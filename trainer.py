import time
import os
import re
import json
import math
from datetime import datetime
import random
import pandas as pd
import numpy as np
import sklearn.metrics
import torch
from torch import nn
from tqdm import tqdm
import sklearn.metrics

import warnings
warnings.filterwarnings("ignore")

import torch

## set seed
def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

## early stop
class EarlyStopping:    
    def __init__(self, save_file, loss_bound=None, patience=5, delta=0):
        """
        save_file(str): save file directory
        patience(int): How long to wait after last time validation loss improved
        delta(float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.save_file = save_file
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.is_stop = False
        self.best_epoch = 0
        self.best_step = 0
        self.loss_bound = loss_bound if loss_bound else np.Inf
        self.start = False
        
    def save_checkpoint(self, metric, model):
        torch.save(model.state_dict(), self.save_file)
        
    def __call__(self, metric, model, loss, increase=True, epoch=0, step=0):
        score = metric
        message=""
        flag = increase*2-1  # True:+1 False:-1
        if loss<self.loss_bound:
            self.start = True
        if self.start:
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(metric, model)            
            elif flag*score < flag*(self.best_score + self.delta):
                self.counter += 1
                #message = f"EarlyStop: {self.counter}/{self.patience} max: {self.best_score:.3f}"
                message = f" E:{self.counter:3d} m:{self.best_score:.3f}"
                if self.counter >= self.patience:
                    self.is_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(metric, model)
                self.counter = 0
                self.best_epoch = epoch
                self.best_step = step
                self.is_stop = False
        print(message, end="")



events5 = ['ferguson', 'sydneysiege', 'germanwings', 'ottawashooting', 'charliehebdo']
events9 = ['sydneysiege', 'ottawashooting', 'charliehebdo', 'ferguson', 'germanwings', 
           'putinmissing', 'gurlitt', 'prince', 'ebola']

## trainer
class Trainer:    
    def __init__(self):
        # 初始化训练环境 创建存储文件夹等
        if not os.path.exists("./save"):
            os.mkdir("./save")

    def _set_ratio_verify(self, data, cids):
        # 计算verify任务的比例
        temp = data.loc[cids]["verify"]
        class_count = temp.value_counts().to_dict()
        ratio = [int(class_count.get(i, 0)) for i in range(3)]
        ratio_str = "/".join([str(item) for item in ratio])
        return ratio, ratio_str
    
    def _set_ratio_stance(self, data, cids):
        # 计算stance任务的比例
        temp = data[data["cid"].isin(cids)]["stance"]
        class_count = temp.value_counts().to_dict()
        ratio = [int(class_count.get(i, 0)) for i in range(4)]
        ratio_str = "/".join([str(item) for item in ratio])
        return ratio, ratio_str              
    
    def _resample_cids(self, data, cids1, cids2, mode="max"):
        data1 = data.loc[cids1]
        class_cids = [data1[data1["verify"]==i]["cid"].tolist() for i in range(3)]
        max_cid_num = max([len(item) for item in class_cids])
        min_cid_num = min([len(item) for item in class_cids])
        random.seed(0)
        class_cids_new = [item*(max_cid_num//len(item))+random.sample(item, max_cid_num%len(item)) for item in class_cids]
        class_cids_new = [x for y in class_cids_new for x in y] # flatten
        return class_cids_new     
    
    def _split_batch(self, indexes, b_size, drop_tail): # indexes:[([23,25], [1,1]), ...]
        index_len = len(indexes)
        if drop_tail:
            index_batch = [indexes[i*b_size: min((i+1)*b_size, index_len)] for i in range(index_len//b_size)]
        else:
            index_batch = [indexes[i*b_size: min((i+1)*b_size, index_len)] for i in range(index_len//b_size+1)]
        return index_batch
    
    @staticmethod
    def get_batch_data(data, cids, args, device):
        cids = list(sorted(cids))
        data_temp = data[data["cid"].isin(cids)].sort_values(by=["cid", "time"])
        index_dict = {str(item[1]):item[0] for item in enumerate(data_temp["mid"])}
        data_temp["parent"] = data_temp["pid"].apply(lambda x:index_dict[x])
        data_temp["child"] = data_temp["mid"].apply(lambda x:index_dict[x])
        ## x input
        x = data_temp["content_glove"].tolist()
        max_sent_len = max([len(sent) for sent in x])
        x = [sent+[0]*(max_sent_len-len(sent)) for sent in x] # padding
        x = torch.Tensor(x).long().to(device)
        x_mask = (x>0)*1
        ## other input
        cas_sent_num = [data_temp["cid"].value_counts().loc[cid] for cid in cids]
        cas_indexes = [data_temp[data_temp["cid"]==cid]["child"].tolist() for cid in cids]
        adjacency_list = torch.Tensor(data_temp[["parent", "child"]].values.tolist()).long().to(device)
        node_order = torch.Tensor(data_temp["node_order"].tolist()).long().to(device)
        edge_order = torch.Tensor(data_temp["edge_order"].tolist()).long().to(device)
        yv = torch.Tensor(data_temp["verify"][data_temp["edge_order"]==-1].tolist()).long().to(device)
        ys = torch.Tensor(data_temp["stance"].tolist()).long().to(device)
        return x, x_mask, adjacency_list, node_order, edge_order, yv, ys
    
    @staticmethod
    def convert_y(yt, yp):
        # 将one-hot预测的y转换为
        yp = np.array(torch.argmax(yp, dim=1).tolist())
        yt = np.array(yt.tolist())
        return yt, yp
    
    def _eval_save_memory(self, model, data, indexes, args, device, b_size=32):
        model.eval()
        index_batch = [indexes[i*b_size: min((i+1)*b_size, len(indexes))] for i in range((len(indexes)-1)//b_size+1)]
        y1, y2, y3, y4 = [], [], [], []
        for step in range(len(index_batch)):
            batch_data = self.get_batch_data(data, index_batch[step], args, device)
            with torch.no_grad():
                y_list, loss_list, z = model(batch_data)
            yvp, yvt, ysp, yst = y_list
            y1.append(yvt)
            y2.append(yvp)
            y3.append(yst)
            y4.append(ysp)
        y1 = torch.cat(tuple(y1))
        y2 = torch.cat(tuple(y2))
        y3 = torch.cat(tuple(y3))
        y4 = torch.cat(tuple(y4))
        return y1, y2, y3, y4
    
    
    def _eval_result(self, y1, y2, loss_func=None): # y1-真实标签 y2-预测的one-hot标签
        class_num = y2.shape[1]
        loss = loss_func(y2, y1).item() if loss_func else 0
        # 转换为np.array
        y2 = np.array(torch.argmax(y2, dim=1).tolist())
        y1 = np.array(y1.tolist())    
        # 除去为-1的空值标签
        y2 = y2[y1>=0] 
        y1 = y1[y1>=0]
        acc = sklearn.metrics.accuracy_score(y1, y2)
        macroF = sklearn.metrics.f1_score(y1, y2, average='macro')
        microF = sklearn.metrics.f1_score(y1, y2, average='micro')
        classF = sklearn.metrics.f1_score(y1, y2, average=None).tolist()
        classP = sklearn.metrics.precision_score(y1, y2, average=None).tolist()
        classR = sklearn.metrics.recall_score(y1, y2, average=None).tolist()
        ytrue = "/".join([str(np.sum(y1==i)) for i in range(class_num)])
        ypred = "/".join([str(np.sum(y2==i)) for i in range(class_num)])
        info = {"loss":loss,"acc":acc,"macroF":macroF,"microF":microF,
                "classF":classF,"classP":classP,"classR":classR,
                "ytrue":ytrue,"ypred":ypred}
        return info
    
    
    def generate_fold(self, data, args):        
        # 根据任务判断需要的数据来源
        folds = []
        data_temp = data[data["verify"]>=0]
        for test_event in events5:
            train_events = [event for event in events5 if event!=test_event]
            train_cids = pd.unique(data_temp[data_temp["event"].isin(train_events)]["cid"]).tolist()
            random.seed(2020)
            random.shuffle(train_cids)
            test_cids = pd.unique(data_temp[data_temp["event"]==test_event]["cid"]).tolist()
            train_cids = self._resample_cids(data, train_cids, test_cids, mode="max")
            cross_cids = test_cids 
            train_ratiov, cross_ratiov, test_ratiov = tuple([self._set_ratio_verify(data, item)[1] 
                                                           for item in [train_cids, cross_cids, test_cids]])
            train_ratios, cross_ratios, test_ratios = tuple([self._set_ratio_stance(data, item)[1] 
                                                           for item in [train_cids, cross_cids, test_cids]])
            folds.append({"train":(train_events, train_cids, train_ratiov, train_ratios),
                          "cross":(test_event, cross_cids, cross_ratiov, cross_ratios),
                          "test":(test_event, test_cids, test_ratiov, test_ratios)})
        return folds
    
    
    def train(self, model, data, train_, cross_, device, args, savefilev, savefiles): 
        print(f"save at {savefilev} & {savefiles}")
        patience = int((((len(train_)-1)//args.batch_size+1)//args.step_interval+1)*args.EPOCH_patience)
        earlystopv = EarlyStopping(save_file=savefilev, patience=patience, delta=0.001)
        earlystops = EarlyStopping(save_file=savefiles, patience=patience, delta=0.001)
        train_info = {}
        optimizer = torch.optim.Adam([{"params":model.sent_model.parameters(), "lr":args.lr},
                                      {"params":model.interaction_model.parameters(), "lr":args.lr},
                                      {"params":model.evolution_model.parameters(), "lr":args.lr},
                                      {"params":model.stance.parameters(), "lr":args.lr_stance},
                                      {"params":model.verify.parameters(), "lr":args.lr_verify},])
        loss_func = nn.CrossEntropyLoss(ignore_index=-1)
        
        for epoch in range(args.EPOCH):            
            random.seed(0)
            random.shuffle(train_)            
            train_batch = self._split_batch(train_, args.batch_size, drop_tail=True)
            ## train
            model.train()
            for step in range(len(train_batch)):
                batch_data = self.get_batch_data(data, train_batch[step], args, device)  # (x1, x2, y, w_size, w_num)
                y_list, loss_list, z = model(batch_data) # losses: (loss_vae, re_loss, kl)                 
                yvp, yvt, ysp, yst = y_list # 多任务的y
                lossv = loss_func(yvp, yvt)
                losss = loss_func(ysp, yst)
                lossd, loss_vae, loss_kl = loss_list
                loss = args.l1*lossv + args.l2*losss + args.l3*lossd
                loss.backward()
                optimizer.step()
                ltrain = [lossv, losss, lossd, loss_vae, loss_kl]
                
                if epoch<args.EPOCH_min:
                    if step==0:
                        model.eval()
                        y1, y2, y3, y4 = self._eval_save_memory(model, data, cross_, args, device) # evaluate save memory           
                        vinfo = self._eval_result(y1, y2, loss_func)
                        sinfo = self._eval_result(y3, y4, loss_func)
                        lossv,losss,accv,accs,macroFv,macroFs = vinfo["loss"],sinfo["loss"],vinfo["acc"],sinfo["acc"],vinfo["macroF"],sinfo["macroF"]
                        #print(f"{epoch:-2d}/{step:2d}|train {ltrain[0]:.3f} {ltrain[1]:.3f} {ltrain[2]:.3f}|cross acc: {accv:.3f} {accs:.3f} maF: {macroFv:.3f} {macroFs:.3f} loss: {lossv:.4f} {losss:.4f}", end="\n")
                        print(f"{epoch:-2d}/{step:2d}|train {ltrain[0]:.3f} {ltrain[1]:.3f} {ltrain[3]:.3f} {ltrain[4]:.3f}|cross maF: {macroFv:.3f} {macroFs:.3f} loss: {lossv:.4f} {losss:.4f}", end="\n")
                        model.train()
                else:
                    ## evaluate
                    if step%args.step_interval==0:
                        model.eval()
                        y1, y2, y3, y4 = self._eval_save_memory(model, data, cross_, args, device) # evaluate save memory           
                        vinfo = self._eval_result(y1, y2, loss_func)
                        sinfo = self._eval_result(y3, y4, loss_func)
                        lossv,losss,accv,accs,macroFv,macroFs = vinfo["loss"],sinfo["loss"],vinfo["acc"],sinfo["acc"],vinfo["macroF"],sinfo["macroF"]
                        #print(f"{epoch:-2d}/{step:2d}|train {ltrain[0]:.3f} {ltrain[1]:.3f} {ltrain[2]:.3f}|cross acc: {accv:.3f} {accs:.3f} maF: {macroFv:.3f} {macroFs:.3f} loss: {lossv:.4f} {losss:.4f}", end="")
                        print(f"{epoch:-2d}/{step:2d}|train {ltrain[0]:.3f} {ltrain[1]:.3f} {ltrain[3]:.3f} {ltrain[4]:.3f}|cross maF: {macroFv:.3f} {macroFs:.3f} loss: {lossv:.4f} {losss:.4f}", end="")
                        model.train()
                        earlystopv(macroFv, model, lossv, increase=True, epoch=epoch, step=step)
                        earlystops(macroFs, model, losss, increase=True, epoch=epoch, step=step)
                        print()
                        if earlystopv.is_stop and earlystops.is_stop:
                            print(f"verify ealy stopped at epoch {earlystopv.best_epoch} step {earlystopv.best_step}!")
                            print(f"stance early stopped at epoch {earlystops.best_epoch} step {earlystops.best_step}!")
                            break
            if earlystopv.is_stop and earlystops.is_stop:
                break
        train_info = {}
        train_info["verify"] = {"epoch":earlystopv.best_epoch, "step":earlystopv.best_step}
        train_info["stance"] = {"epoch":earlystops.best_epoch, "step":earlystops.best_step}        
        return train_info
    
    def _show_table(self, rows):
        for row in rows:
            for item in row:
                if isinstance(item, str) or isinstance(item, int):
                    print(item, end="\t")
                if isinstance(item, float):
                    print(f"{item:.3f}", end="\t")
            print()
    
    def test(self, model, data, test_, device, args, savefilev, savefiles, train_info):
        print(f"\ntesting... load model from {savefilev} & {savefiles}")
        test_info = {}
        statev = torch.load(savefilev)
        model.load_state_dict(statev)
        model.eval()
        y1, y2, y3_, y4_ = self._eval_save_memory(model, data, test_, args, device)
        states = torch.load(savefiles)
        model.load_state_dict(states)
        model.eval()
        y1_, y2_, y3, y4 = self._eval_save_memory(model, data, test_, args, device)
        
        vinfo = self._eval_result(y1, y2)
        vinfo.update(train_info["verify"])
        test_info["verify"] = vinfo
        sinfo = self._eval_result(y3, y4)
        sinfo.update(train_info["stance"])
        test_info["stance"] = sinfo
        rows = [["task", "acc","macroF","classF"]]
        rows.append(["verify", vinfo["acc"], vinfo["macroF"]]+vinfo["classF"])
        rows.append(["stance", sinfo["acc"], sinfo["macroF"]]+sinfo["classF"])
        self._show_table(rows)
        return test_info
    
    def show_info(self, events, info_lists, fields):
        for task in ["verify", "stance"]:
            print(f"=== {task} ===")
            info_list = [item[task] for item in info_lists]
            temp = [info_list[0][field] for field in fields]
            temp = [len(item) if isinstance(item, list) else 1 for item in temp]
            title = "\t".join(["test"]+[fields[i]+"\t"*(temp[i]-1) for i in range(len(fields))])
            print(f"\n{title}")
            values = []
            for info in info_list:
                temp = []
                for field in fields:
                    if isinstance(info[field], list):
                        temp.extend(info[field])
                    else:
                        temp.append(info[field])
                values.append(temp)
            values_avg = [[values[i][j] for i in range(len(values))] for j in range(len(values[0]))]
            values_avg = [f"{np.mean(item):.3f}" if not isinstance(item[0], str) else "-" for item in values_avg]
            for i in range(len(events)):
                value = values[i]
                print("\t".join([events[i]]+[f"{item:.3f}" if isinstance(item, float) else str(item) for item in value]))
            print("\t".join(["Avg-"]+[item for item in values_avg])) 
        
    def start(self, sent, inter, evolution, stance, verify, hierarchy, data, pretrained_weight, args, device, fold_num=None):        
        time_start = datetime.now()           
        # 生成fold包含的index
        print("*** spliting folds")
        folds = self.generate_fold(data, args)  # [{"train":(fold_name, cids_list)},...]
        # 对不同fold的数据进行训练
        events, results = [], []
        print("\n*** training")
        folds = folds if fold_num is None else folds[fold_num:fold_num+1]
        for fold in folds:
            events.append(fold["test"][0][:5])
            time_start_fold = datetime.now().strftime("%Y%m%d-%H%M%S")
            savefilev = f"./save/{time_start_fold}_verify.pkl"
            savefiles = f"./save/{time_start_fold}_stance.pkl"
            # 显示数据比例
            print(f"\n--- \ntest on {fold['test'][0]}({fold['test'][2]}, {fold['test'][3]})")
            print(f"train ({fold['train'][2]}, {fold['train'][3]})")
            # 获取train cross test涉及的cid
            train_, cross_, test_ = tuple([fold[set_name][1] for set_name in ["train", "cross", "test"]])
            # 初始化模型信息  
            set_seed(0)
            model = hierarchy(sent, inter, evolution, stance, verify, args, pretrained_weight)
            model.to(device)
            # 训练
            train_info = self.train(model, data, train_, cross_, device, args, savefilev, savefiles)
            # 测试
            test_info = self.test(model, data, test_, device, args, savefilev, savefiles, train_info)
            results.append(test_info)   
        show_fields = ["epoch","step","acc","macroF","classF","ytrue","ypred"]
        self.show_info(events, results, show_fields)
        time_end = datetime.now()
        print("\nstart at ", time_start)
        print("end at   ", time_end)
        return None