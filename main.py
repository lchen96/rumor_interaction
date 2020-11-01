import torch

from dataloader import DataLoader
from model import GloveMean, DVAE, Evolution, Stance, Verify, Hierarchy
from trainer import Trainer

import argparse


if __name__=="__main__":
    ## data
    parser = argparse.ArgumentParser(description='modeling evolution of message interaction for rumor resolution')
    parser.add_argument('--rawdata_file', default="./data/pheme.csv",
                        help='path of rawdata', type=str)
    parser.add_argument('--abbreviation_file', default="./data/abbreviation.json",
                        help='path of abbreviation json file', type=str)
    parser.add_argument('--vocab_size', default=0,
                        help='vocabulary size (0/10000/...)', type=int)
    parser.add_argument('--pretrain_file', default="/remote-home/source/Social/Datasets/embedding/glove/glove.6B.300d.txt",
                        help='pretrained weight of glove embedding', type=str)
    parser.add_argument('--embedding_size', default=300,
                        help='embedding size', type=int)
    ## model parameter: sentence representation
    parser.add_argument('--sent_size', default=300,
                        help='size of sentenct', type=int)
    ## model parameter: dvae
    parser.add_argument('--K', default=4,
                        help='dimension of latent variable', type=int)
    parser.add_argument('--M', default=4,
                        help='quantity of latent variable', type=int)
    parser.add_argument('--alpha', default=1,
                        help='tradeof between expectation and kl divergence', type=int)
    parser.add_argument('--temperature', default=10,
                        help='temperature to control gumbel-softmax', type=int)
    parser.add_argument('--inter_size', default=300,
                        help='size of interaction representation', type=int)
    ## model parameter: evolution
    parser.add_argument('--num_layers', default=1,
                        help='layer of bilstm', type=int)
    parser.add_argument('--hidden_size', default=200,
                        help='hidden size of BiLSTM', type=int)
    parser.add_argument('--dropout', default=0.5,
                        help='drop out rate of dense layer', type=float)
    ## experiments
    parser.add_argument('--gpu', default=0,
                        help='serial number of gpu', type=int)
    parser.add_argument('--batch_size', default=32,
                        help='batch size', type=int)
    parser.add_argument('--EPOCH', default=60,
                        help='running epoch', type=int)
    parser.add_argument('--EPOCH_min', default=0,
                        help='running epoch without fine evaluation', type=int)
    parser.add_argument('--step_interval', default=1,
                        help='step interval for evaluation', type=int)
    parser.add_argument('--EPOCH_patience', default=40,
                        help='patience for earlystop', type=int)
    parser.add_argument('--lr', default=1e-5,
                        help='learning rate of other layers', type=float)
    parser.add_argument('--lr_stance', default=1e-4,
                        help='learning rate of stance classification', type=float)
    parser.add_argument('--lr_verify', default=1e-5,
                        help='learning rate of verification', type=float)
    parser.add_argument('--l1', default=0.5,
                        help='loss weight of verification', type=float)
    parser.add_argument('--l2', default=0.5,
                        help='loss weight of stance classification', type=float)
    parser.add_argument('--l3', default=0.2,
                        help='loss weight of dvae', type=float)
    
    ## initialize parameters
    args = parser.parse_known_args()[0]
    ## assign device
    device = torch.device(f"cuda: {args.gpu}" if torch.cuda.is_available() else "cpu")
    ## load data
    data_loader = DataLoader(args)
    data, pretrained_weight = data_loader.load()
    ## training
    trainer = Trainer()
    trainer.start(GloveMean, DVAE, Evolution, Stance, Verify, Hierarchy, data, pretrained_weight, args, device)