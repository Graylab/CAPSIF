#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 16:54:07 2022

@author: sudhanshu
Trains the CAPSIF:G model on the dataset
Note, must be run after preprocess.py has been performed

Usage: python train.py
Returns: prediction model into "models_DL/capsif_g_model.pth.tar"

"""

import egnn.egnn as eg
import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from dataset import graph_data_loader, model_train, calc_loss, dice_loss
# from torchsummary import summary

TRAIN_DIR = "../dataset/train/g_npz/"
VAL_DIR = "../dataset/val/g_npz/"
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
BATCH_SIZE = 1
NUM_EPOCHS = 1000
NUM_WORKERS = 0
PIN_MEMORY = True
LOAD_MODEL = 0
LAYERS = 32  # training layers
MAX_ROTATION = 180 # Degree
MAX_PIXEL_TRANSLATION = 3 #count

# Dummy parameters
batch_size = 1
n_nodes = 4
n_feat = 29
x_dim = 1
#DEVICE = 'cuda'


model = eg.EGNN2(in_node_nf=n_feat, hidden_nf=n_feat*2, out_node_nf=1, in_edge_nf=1,
                 n_layers=8, attention=1,normalize=1).to(DEVICE)

# Run EGNN
#h, x = model(h, x, edges, edge_attr)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scaler = torch.cuda.amp.GradScaler()

model.train()
loss_chart=[]

train_loader,val_loader = graph_data_loader(train_dir=TRAIN_DIR, val_dir=VAL_DIR
                                            , batch_size=1)
loss_fn = dice_loss
best_acc=0

x = []
train_loss = []
val_loss = [];

for epoch in range(1000):
    x.append(epoch)

    print(epoch)
    step_loss = model_train(train_loader, model, optimizer, loss_fn, scaler, DEVICE=DEVICE)

    train_loss.append(step_loss)

    step_acc = calc_loss(val_loader, model, DEVICE=DEVICE)
    val_loss.append(step_acc)

    if step_acc > best_acc:
        best_acc = step_acc.item()



        checkpoint = {
            "state_dict":model.state_dict(),
            "optimizer":optimizer.state_dict(),
            "epoch": epoch,
            "best": best_acc

        }
        torch.save(checkpoint, "models_DL/capsif_g_model.pth.tar" )

    print("Val DICE:%5.2f" % step_acc.item(), ", Best DICE:%5.2f" % best_acc)

    plt.plot(x,train_loss,'r',label="train")
    plt.plot(x,val_loss,'b',label="Val")
    plt.ylim([0,1])
    plt.savefig("./loss.png",transparent=True,dpi=300)
