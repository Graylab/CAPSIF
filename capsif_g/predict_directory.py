#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 16:54:07 2022

@author: sudhanshu
Runs Prediction on input directory from command line

Usage: python predict_directory.py [pdb_directory: default '../sample_dir' ][model_directory: default './models_DL/cb_model.pth.tar']
Returns: PDB_NAME: predicted residues ([ResidueNumber].[ChainName])
"""

import egnn.egnn as eg
import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from dataset import graph_data_loader, model_train, calc_loss, dice_loss, predict_directory
import sys

n = len(sys.argv)
#print(n,sys.argv)

PREDICT_DIR = "../sample_dir/"
if (n >= 2):
    PREDICT_DIR = sys.argv[1].strip() + "/"

model_dir = "./models_DL/cb_model.pth.tar"
if (n > 2):
    model_dir = sys.argv[2].strip()
print("Using model: " + model_dir)

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


#model_dir = "./models_DL/capsif_g_model.pth.tar"

# Dummy parameters
batch_size = 1
n_nodes = 4
n_feat = 29
x_dim = 1

loss_fn = dice_loss
best_acc=0

#print("loaded")
with torch.no_grad():

    predict_directory(model_dir,PREDICT_DIR,cutoff=0.5)
