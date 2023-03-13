#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 16:01:47 2022

@author: sudhanshu
"""
import numpy as np
import os
import shutil
train_percent = 72
val_percent = 8  #(of 80)
test_percent = 20

# np.random.seed(3123)

uniq_pdbs = "../pdb_chains/dataset/dataset_voxel/uniq_pdb_chains.txt"
randomized_files = "../pdb_chains/dataset/dataset_voxel/"

rand_file_names =[]
for i in os.listdir(randomized_files+"protein/"):
    if i.endswith(".npz"):
        rand_file_names.append(i[:-4])



pdb_names = []

fid = open(uniq_pdbs,"r")
pdb_list = fid.readlines()
fid.close()

for i in pdb_list:
    pdb_names.append(i.split()[0])
    
randomize_number = np.random.permutation(len(pdb_names))


val_data_len = int(len(pdb_names)*val_percent/100)
train_data_len = int(len(pdb_names)*train_percent/100) 
#test_data_len  = int(len(pdb_names)*test_percent/100)






train_pdb_ids = [ pdb_names[i] for i in randomize_number[:train_data_len] ]
val_pdb_ids = [ pdb_names[i] for i in randomize_number[train_data_len:train_data_len + val_data_len] ]
test_pdb_ids = [ pdb_names[i] for i in randomize_number[(train_data_len+val_data_len):] ]

data_train_output = "../dataset/dataset_random_for_train_and_val/"

level_1_dir = ['train','val', 'test']
level_2_dir = ['mask', 'protein']

for i in level_1_dir:
    if not os.path.exists(data_train_output + i):
        os.mkdir(data_train_output+i)
        
    for j in level_2_dir:
        if not os.path.exists(data_train_output + i +"/" +j):
            os.mkdir(data_train_output + i +"/" +j)
        
        

#making_train

fid_train = open(data_train_output +"train_list.txt","w+")
fid_test = open(data_train_output +"test_list.txt","w+")
fid_val = open(data_train_output +"val_list.txt","w+")

for i in train_pdb_ids:
    fid_train.write(i+"\n")
    
for i in test_pdb_ids:
    fid_test.write(i+"_000_\n") 

for i in val_pdb_ids:
    fid_val.write(i+"_000_\n") 



    
fid_test.close()
fid_train.close()
fid_val.close()  
    


