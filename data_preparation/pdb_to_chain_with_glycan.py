#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 17:56:21 2022

This code identifies and creates pdb files of chains and interacting glycans.
It workd on multi chain pdb files given in a directory. 
Note: Input directory has only glycan interacting pdbs.
//

@author: sudhanshu
"""
from mpl_toolkits.mplot3d import Axes3D
from pyrosetta import *
from pyrosetta.teaching import *
from pyrosetta.rosetta.core.pose import *
import os
from data_utils import ( done_data_recorder, 
                        keep_chains_and_interaction)

init('-include_sugars  -ideal_sugars -ignore_unrecognized_res -renumber_pdb -ignore_zero_occupancy false  -alternate_3_letter_codes pdb_sugar -maintain_links -auto_detect_glycan_connections  -mute all')



def main():
    
    seq_chains = '0ABCDEFGHIJKLM'  # for int to chain converter
    
    #RCSB chain clusters for glycan interaction pdb chains    
    cluster_file = "./data_files/clusters.txt"
    fid = open(cluster_file,'r')
    clusters = fid.readlines()
    fid.close()
    
    
    #PATH pdb chain files
    dir_path = "/home/sudhanshu/HDD2/projects2/utils/glyco_data/all_pdbs_from_PDB_only/usable_pdb_for_motif_finding/2022_dataset_for_DL/pdb_selected_plus_lectins/chain_wise/"

    out_path = dir_path + "chains/"
    
    # file treatmet tracker for multicore filtering
    done_files = done_data_recorder(dir_path+"done_chains.txt",str)
    
    
    for i in clusters:
        if i.startswith("#"):
            continue
        
        i_sp = i.split()
        
        
        
        for j in i_sp:
            if done_files.check_val_exist(j):
                continue
            done_files.add_val(j)
            chain = int(j.split("_")[1])
            pdb_f = j.split("_")[0]
            
            in_pdb = dir_path + pdb_f +".pdb"
            out_pdb = out_path + j + ".pdb"
            
            if not os.path.exists(in_pdb):
                continue
            
            try:
                pose = pose_from_pdb(in_pdb)
                keep_chains_and_interaction(pose, seq_chains[chain], out_pdb)
            except RuntimeError:
                print(in_pdb + " can't be read.")
                continue
                
            
if __name__ == "__main__":
    main()
        
        
        
    

