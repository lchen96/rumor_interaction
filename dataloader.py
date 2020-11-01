import os
import pickle
import re
import json

## loading nltk packages for preprocessing text
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

## tools for preprocess
tweet_tokenizer = TweetTokenizer()
word_lemmatizer = WordNetLemmatizer()
stop_words = list(set(stopwords.words('english')))
stop_words = []

## loading dict of collected word abbreviation
with open("./data/abbreviation.json", "r", encoding="utf-8") as f:
    abbre_dict = dict(json.load(f))

class GloveTokenizer:
    
    def load_vocab_weight(self, vocab, vocab_size, pretrain_file, embedding_size):
        print("loading pretrained glove weight...")
        vocab_size = len(vocab) if not vocab_size else vocab_size # vocab_size 0/None: all 
        vocab_index_dict = {"<PAD>":0,"<UNK>":1}
        random_boud = np.sqrt(3./embedding_size)
        init_pretrained_weights = np.zeros(shape=(vocab_size, embedding_size), dtype=np.float32)
        init_pretrained_weights[1] = np.random.uniform(-random_boud, random_boud, embedding_size)
        embed_dict = {} # {word:vector}
        with open(pretrain_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.split(" ")
                assert len(line[1:]) == embedding_size
                embed_dict[line[0].lower()] = line[1:]
        count = 0
        embed_keys = embed_dict.keys()
        word_index = 2
        for key in vocab[2:]:
            if key in embed_keys:
                init_pretrained_weights[word_index] = embed_dict[key]
                vocab_index_dict[key] = word_index
                word_index+=1
            else:
                count += 1
            if word_index>=vocab_size:
                break
        init_pretrained_weights = init_pretrained_weights[:word_index]
        print(f"mapped {count+word_index} words, {count} words not in pretrained glove embedding")
        pretrained_weight = init_pretrained_weights
        return vocab_index_dict, pretrained_weight
    
    def __init__(self, vocab, vocab_size, pretrain_file, embedding_size):
        vocab_index_dict, pretrained_weight = self.load_vocab_weight(vocab, vocab_size, pretrain_file, embedding_size)
        self.vocab_index_dict = vocab_index_dict
        self.pretrained_weight = pretrained_weight  
    
    def tokenize(self, sentence):
        sentence = sentence.split(" ")
        return sentence
    
    def convert_tokens_to_ids(self, sentence):
        # convert words not in glove into <UNK>
        ids = [self.vocab_index_dict.get(word, 1) for word in self.tokenize(sentence)] 
        return ids
    
import time
import pandas as pd
from collections import Counter  # count words
import numpy as np
from tqdm import tqdm

class DataLoader:
    def __init__(self, args):
        ## initialize 'preprocess' directory
        if 'preprocess' not in os.listdir():
            os.mkdir('preprocess')
        self.rawdata_file = args.rawdata_file
        self.vocab_size = args.vocab_size
        self.pretrain_file = args.pretrain_file
        self.embedding_size = args.embedding_size
        
    
    def clean_text(self, sentence): 
        sentence = re.sub("[hH]ttp\S*", "", sentence)    # remove url
        sentence = re.sub("@\S*","", sentence)           # remove @
        sentence = re.sub("#", "", sentence)             # remove #
        sentence = re.sub( r"([A-Z][a-z]*)", r"\1", sentence) # split with capitalized letter
        sentence = re.sub("[0-9]", "", sentence)         # remove numbers
        sentence = sentence.lower()                      # convert into lowercase
        sentence = " ".join([abbre_dict.get(word, 0) if abbre_dict.get(word, 0) else word 
                             for word in re.split("([\.\+\-\?\"\\,!/\s])", sentence)]) # split with .+-?"\,!/"
        sentence = " ".join(tweet_tokenizer.tokenize(sentence))
        sentence = " ".join([word_lemmatizer.lemmatize(word, pos='v') for word in sentence.split()])
        sentence = " ".join([word for word in sentence.split() if word not in stop_words])
        sentence = " ".join([abbre_dict.get(word, 0) if abbre_dict.get(word, 0) else word 
                             for word in sentence.split()]) # split with .+-?"\,!/"
        if not sentence:
            sentence = "null"
        return sentence        
    
    def preprocess_data(self, df):
        print("preprocessing data...")
        ## convert index into string format to avoid int/float errors
        df["cid"] = df["cid"].apply(str)
        df["mid"] = df["mid"].apply(str)
        df["pid"] = df["pid"].apply(str)
        df.index = df["mid"]
        df.index.name = None
        ## ensure every parent can be find in the dataset【TODO】 处理无法形成父子对的id 两种方式
        cid_mid_dict = {cid:df[df["cid"]==cid]["mid"].tolist() for cid in pd.unique(df["cid"])}
        lost_parent_num = 0
        for index, row in tqdm(df.iterrows()):
            if row["pid"] not in cid_mid_dict[row["cid"]]:
                df.loc[index, "pid"] = row["cid"] # replace the lost parent with source
                lost_parent_num += 1
        print(f"{lost_parent_num} parent cannot be found in raw data")
        ## clean text
        df["content_clean"] = df["content"].apply(self.clean_text)
        ## obtain vocab set
        counter = Counter()
        for sentence in df["content_clean"].tolist():
            counter.update(re.split(" ", sentence))
        word_freq = sorted(counter.items(), key=lambda x:x[1], reverse=True)
        vocab = list(list(zip(*word_freq))[0])
        vocab = ["<PAD>", "<UNK>"] + vocab
        print(f"vocab size (all): {len(vocab)}")
        ## initialize glove tokenizer
        glove_tokenizer = GloveTokenizer(vocab=vocab, 
                                         vocab_size=self.vocab_size, 
                                         pretrain_file=self.pretrain_file, 
                                         embedding_size=self.embedding_size)
        ## convert words into glove ids
        df["content_glove"] = df["content_clean"].apply(glove_tokenizer.convert_tokens_to_ids)
        pretrained_weight = glove_tokenizer.pretrained_weight
        return df, pretrained_weight    
    
    def form_tree_data(self, df): 
        print("forming tree-related information...")
        ## initialize node edge order
        df["node_order"] = 0
        df["edge_order"] = -1
        cids = pd.unique(df["cid"])
        for cid in cids: # mapping all the cascade
            df_temp = df[df["cid"]==cid]
            mids = df_temp["mid"]
            pids = df_temp["pid"]
            if len(mids)>1:
                ## form mid_pid_dict & pid_mid_dict
                mid_pid_dict = {item[0]:item[1] for item in zip(df_temp["mid"], df_temp["pid"])}
                pid_mid_dict = {}
                for mid, pid in mid_pid_dict.items():
                    pid_mid_dict[pid] = pid_mid_dict.get(pid, []) + [mid]
                ## find leaves (without children)
                leaves = [item[0] for item in mid_pid_dict.items() if item[0] not in mid_pid_dict.values()]            
                ## calculate depth of each node
                node_depth = {}
                for mid in mids:
                    depth = 0 if mid==mid_pid_dict[mid] else 1
                    pid = mid_pid_dict[mid]
                    while pid != cid:
                        depth += 1
                        pid = mid_pid_dict[pid]
                    node_depth[mid] = depth
                node_depth = sorted(node_depth.items(), key=lambda x:x[1], reverse=True)
                ## calculate node_order leaves: 0 / others: maximum number of their childern + 1
                node_order = {leaf:0 for leaf in leaves}
                for node, depth in node_depth:
                    if node not in node_order.keys():
                        node_order[node] = max([node_order[child] for child in pid_mid_dict[node] if child!=node])+1
                df.loc[list(node_order.keys()), "node_order"] = list(node_order.values())
                df.loc[mids, "edge_order"] = df.loc[pids]["node_order"].values
                df.loc[cid, "edge_order"] = -1 # -1 as a signal of iteration termination
        return df
    
    def load(self):
        time_start = time.time()
        preprocess_file = f"df_preprocess_vocab_{self.vocab_size}.pkl"
        if preprocess_file in os.listdir("./preprocess/"): # load from saved file
            print("loading preprocess data from saved file...")
            with open(os.path.join("./preprocess", preprocess_file), "rb") as f:
                data = pickle.load(f)
            df, pretrained_weight = data            
        else: # construct from raw data            
            ## read raw data
            df = pd.read_csv(self.rawdata_file, encoding="utf-8", sep=",")
            ## preprocess text
            df, pretrained_weight = self.preprocess_data(df)
            ## form tree information
            df = self.form_tree_data(df)
            ## save preprocessed data
            with open(os.path.join("./preprocess", preprocess_file), "wb") as f:
                pickle.dump((df, pretrained_weight), f)
            print("preprocessed data saved!")
        print(f"--- data loaded! consume {time.time()-time_start:.2f} s")
        return df, pretrained_weight           