#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 15:54:37 2022
Last update: 3/4/2023

@author: sudhanshu

Usage: python train.py
Returns: trained model "./xxmy_checkpoint2A_36_CB_coord.pth.tar" alongside Reports of epochs in Reports/

"""

import torch
import torch.optim as optim

from models import  model_train, UNET_3D
from utils import (load_checkpoint, save_checkpoint, get_loaders, check_accuracy,
                   dice_loss, calc_accuracy, report_file_name,
                   xyz_to_rotate_to_voxelize_to_translate,
                   is_model_params_correct)
import sys
import os
sys.path.append("..")
#required for running it in directory :(
#if running in CAPSIF/ and not CAPSIF/capsif_v/ remove this tag
os.chdir("..")

from settings import config1, check_all_dirs


## Training parameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 20
SAVE_AFTER_EPOCH =10
NUM_EPOCHS = 1000
NUM_WORKERS = 4
PIN_MEMORY = True
LOAD_MODEL = False  #for retraining or extension
TRAIN_DIR = os.path.abspath(config1.dataset_dir + "train/pdb_npz/")
VAL_DIR =   os.path.abspath(config1.dataset_dir + "val/pdb_npz/")
LAYERS = 29  # training layers
MAX_ROTATION = 180 # Degree
MAX_PIXEL_TRANSLATION = 7 #count integer
LAYER_29_TYPE = 2 #CA=0,CB=1,unit Vector=2., CB and unit vector 3 (for layer 32), works for LAYERS = 29
# if using any other layer_29 type change output checkpoint name
RANDOMIZE_COORDS_BY=1.0 # activatde for training set only

def main():

    if not is_model_params_correct():
        return
    model_name = UNET_3D  # We tried many other variants
    best_checkpoint_name = (
    "./capsif_v/models_DL/xxmy_checkpoint_best_36_2A_CACB_vector_coord_I_clean_data_rand10.pth.tar"
    )

    # Model parameter names:
    checkpoint_name ="./capsif_v/xxmy_checkpoint2A_36_CB_coord.pth.tar" #"x.pth.tar"# tempfile


    # checking and making all required directories
    check_all_dirs()

    #local variables
    best_accuracy = 0
    report_file = report_file_name("./capsif_v/Reports", use_previous=LOAD_MODEL)
    if LOAD_MODEL == False:
        fid = open(report_file,"w+")
        fid.write("Epoch Train_loss Test_accuracy\n")
    else:
        fid = open(report_file,"a")

    # PDB.NPZ file reader and transformer class
    data_reader_and_trasnformer = xyz_to_rotate_to_voxelize_to_translate()
    data_reader_and_trasnformer.layers_use = LAYERS
    data_reader_and_trasnformer.layer_29_type = LAYER_29_TYPE
    data_reader_and_trasnformer.max_pixel_translate_per_axis = MAX_PIXEL_TRANSLATION
    data_reader_and_trasnformer.max_rotation_plus_minus = MAX_ROTATION
    data_reader_and_trasnformer.randomize_by = RANDOMIZE_COORDS_BY
    #

    # Data loaders
    #print(TRAIN_DIR,VAL_DIR)
    train_loader, val_loader = get_loaders(TRAIN_DIR, VAL_DIR, BATCH_SIZE,
                                            data_reader_and_trasnformer,
                                            NUM_WORKERS, PIN_MEMORY,
                                            layers=LAYERS, )

    #Model
    current_epoch = 0
    model = model_name(in_channels=LAYERS, out_channels=1).to(DEVICE) # more tthan one output channel loss will be cross entropu
    loss_fn =  dice_loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    if LOAD_MODEL:
        current_epoch, best_accuracy = load_checkpoint(torch.load(checkpoint_name), model, optimizer)
        print(current_epoch)
    check_accuracy(val_loader, model, device=DEVICE)

    #Iterations
    for epoch in range(current_epoch, NUM_EPOCHS):
        step_loss = model_train(train_loader, model, optimizer, loss_fn, scaler,DEVICE=DEVICE)

        # check accuracy validation data
        step_acc = calc_accuracy(val_loader, model, device=DEVICE)

        #saving validation predictions
        #save_predictions_as_masks(val_loader, model, folder="saved_masks/", device=DEVICE)
        fid.write("%5d %8.5f %8.5f\n"  % (epoch+1, step_loss, step_acc))

        #Saving checkpoints
        if (epoch+1)%SAVE_AFTER_EPOCH == 0:
            checkpoint = {
                "state_dict":model.state_dict(),
                "optimizer":optimizer.state_dict(),
                "epoch": epoch,
                "best": best_accuracy

            }
            save_checkpoint(checkpoint, filename=checkpoint_name)
            #load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

            #Saving best validation model
        if step_acc > best_accuracy:
            checkpoint = {
                "state_dict":model.state_dict(),
                "optimizer":optimizer.state_dict(),
                "epoch": epoch,
                "best": best_accuracy

            }
            best_accuracy = step_acc
            save_checkpoint(checkpoint, filename=best_checkpoint_name)

        fid.close()
        fid= open(report_file,"a")

    #save final model
    checkpoint = {
        "state_dict":model.state_dict(),
        "optimizer":optimizer.state_dict(),
        "epoch":current_epoch,
        "best": best_accuracy
    }
    save_checkpoint(checkpoint)

if __name__ == "__main__":
    main()
