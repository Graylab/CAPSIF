#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 14:40:22 2022

This code can be used to filter down all Rosetta Readable pdb files.
It runs on the given directory and moves all unreadable files to 
./unread/.
last update 3/6/2023

@author: sudhanshu
"""
import os

from pyrosetta import *
from pyrosetta.teaching import *
from pyrosetta.rosetta.core.pose import *
import random
import shutil
from data_utils import done_data_recorder

init('-include_sugars  -ideal_sugars -ignore_unrecognized_res -renumber_pdb -ignore_zero_occupancy false  -alternate_3_letter_codes pdb_sugar -maintain_links -auto_detect_glycan_connections  -mute all')


def main(path_pdbs):
    # path_pdbs = "/home/sudhanshu/HDD3/Data/Sudhanshu/pdb_chains/dataset/pdbs/test/"
    unread = path_pdbs + "unread/" # to transfer unreadable files
    
    
    pdbs = []
    for i in os.listdir(path_pdbs):
        if i.endswith('pdb'):
            pdbs.append(i)
    
    
    # making move dir
    if not os.path.exists(unread):
        os.mkdir(unread)
    
    
    # done pdb check 
    done_file = path_pdbs + "done_list.txt"
    done_data = done_data_recorder(done_file,str)


    random.shuffle(pdbs)
    counter = 0
    for i in pdbs:
        counter +=1
        if os.path.exists(done_file):                      
            if done_data.check_val_exist(i):
                print(i+" already done!")
                continue
        
        done_data.add_val(i)        
        pdb_fl = path_pdbs + i
        
        print(counter,"/", len(pdbs))
        try:
            pose_c = pose_from_pdb(pdb_fl)
        except RuntimeError:
            
            if os.path.exists(pdb_fl):
                
                if pdb_fl.endswith("pdb"):
                    print(i + " can't be read.")
                    shutil.move(pdb_fl, unread + i)
        

if __name__ == '__main__':
    pdb_dir = "/home/sudhanshu/HDD3/Data/Sudhanshu/pdb_chains/dataset/pdbs/test/" #Modify This
    main(pdb_dir)
    
    
