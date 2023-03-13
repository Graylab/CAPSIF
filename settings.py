#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 15:30:12 2022

@author: sudhanshu

Usage: File is run to fix some directory settings for CAPSIF:V scripts
"""
import os
import sys

class dot_variable_class:
    def __init__(self):
        pass
    @property
    def all_methods(self):
        return list(self.__dict__.keys())

#Get the directory in a round about way...

def determine_root(file='settings.py'):
    path = os.path.abspath(os.getcwd()) + "/"
    ls = os.listdir(path)
    if file in ls:
        return os.path.abspath(path)
    for ii in range(path.count('/')):
        path = path + "../"
        ls = os.listdir(path)
        if file in ls:
            return os.path.abspath(path)
    print("Could not find path. Please rerun in path/to/CAPSIF directory")
    return '.'

# Give the path of this file followed by "/"
model_root_directory = determine_root() + "/"  #parent directory



# path to npz dataset dir
dataset_dir = model_root_directory + "dataset/"

# SETTINGS FOR CAPSIF_V

# required for training outputs
required_dirs = ["./capsif_v/Reports" ,
                # "capsif_v/saved_masks" ,
                "./capsif_v/models_DL"]


config1 = dot_variable_class()
config1.sys_path = sys.path
config1.sys_path.insert(1,'.')


config1.dataset_dir = dataset_dir
config1.model_root_directory = model_root_directory

# default model names only type 2 provided [used in prediction_utils.py]
config1.model_for_test_and_prediction_type0 = (
    "capsif_v/models_DL/my_checkpoint_best_36_2A_CA_coord_I_clean_data.pth.tar")  #CA
config1.model_for_test_and_prediction_type1 = (
    "capsif_v/models_DL/my_checkpoint_best_36_2A_CB_coord_I_clean_data.pth.tar")  #CB
config1.model_for_test_and_prediction_type2 = (
    "capsif_v/models_DL/my_checkpoint_best_36_2A_CACB_vector_coord_I_clean_data.pth.tar") #CACB_unit_vector
config1.model_for_test_and_prediction_type3 = (
    "capsif_v/models_DL/my_checkpoint_best_36_2A_CA_CB_coord_I_clean_data.pth.tar") #CACB_cordinates




def check_all_dirs ():
    for i in required_dirs:
        full_dir = model_root_directory + i
        if not os.path.exists(full_dir):
            os.mkdir(full_dir)

