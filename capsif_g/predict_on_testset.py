#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 16:54:07 2022

@author: sudhanshu
Runs Prediction on Test Set

Usage: python predict_on_testset.py [model_directory: optional]
Returns: Dice scores, TP,FP,FN,TN for each pdb in Test directory
"""

import egnn.egnn as eg
import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from dataset import graph_data_loader, model_train, calc_loss, dice_loss, predict_for_all_test

import sys
n = len(sys.argv)

model_dir = "./models_DL/cb_model.pth.tar"
if (n > 1):
    model_dir = sys.argv[1].strip()
print("Using model: " + model_dir)

TRAIN_DIR = "../dataset/train/"
VAL_DIR = "../dataset/val/"
TEST_DIR = "../dataset/test/"
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
BATCH_SIZE = 1
#NUM_EPOCHS = 1000
NUM_WORKERS = 0
PIN_MEMORY = True
LOAD_MODEL = 1
LAYERS = 32  # training layers
MAX_ROTATION = 180 # Degree
MAX_PIXEL_TRANSLATION = 3 #count


# Dummy parameters
batch_size = 1
n_nodes = 4
n_feat = 29
x_dim = 1

loss_fn = dice_loss
best_acc=0

#print("loaded")
with torch.no_grad():

    predict_for_all_test(model_dir,TEST_DIR + "g_npz/",cutoff=0.5)
