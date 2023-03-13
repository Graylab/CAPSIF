#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 12:00:41 2022
'This code a chain from the pdb chain cluster with maximum glycan interactions'
@author: sudhanshu
"""

from pyrosetta import *
from pyrosetta.teaching import *
from pyrosetta.rosetta.core.pose import *
import os
import shutil
from data_utils import done_data_recorder

init('-include_sugars  -ideal_sugars -ignore_unrecognized_res -renumber_pdb -ignore_zero_occupancy false  -alternate_3_letter_codes pdb_sugar -maintain_links -auto_detect_glycan_connections  -mute all')




def protein_and_carb_size(pose):
    # To identify size of protein in a chain and interacting carb size
    seq1 = pose.sequence()
    protein_len1 = len(seq1) - seq1.count('Z')
    
    carb_res_pre = [i+1 for i,j in enumerate(seq1) if j =="Z" and pose.residue(i+1).is_carbohydrate()] # i+1 as residue starts from 1
    carb_res = []
    for i in carb_res_pre:
        if pose.glycan_tree_set().get_tree_root_of_glycan_residue(i) == 0:
            carb_res.append(i)    
    
    carbs_in_res = len(carb_res)
    
    return protein_len1, carbs_in_res





def main():
    
    cluster_file ="/home/sudhanshu/HDD2/projects2/voxel_type_pc_interaction/data_preparation/data_files/clusters.txt"

    path1 = "/home/sudhanshu/HDD2/projects2/utils/glyco_data/all_pdbs_from_PDB_only/usable_pdb_for_motif_finding/2022_dataset_for_DL/pdb_selected_plus_lectins/chain_wise/chains/"
    path2 = "/home/sudhanshu/HDD2/projects2/utils/glyco_data/all_pdbs_from_PDB_only/usable_pdb_for_motif_finding/2022_dataset_for_DL/pdb_selected_plus_lectins/chain_wise/chains/clustered/"


    done_file = path1 + "/done_cluster.txt"
    done_pdbs = path1 + "/done_cluster_pdbs.txt"


    clusters = open(cluster_file,"r").readlines()
    
    #for cluster checked record
    done_file_handle = done_data_recorder(done_file,int) 
    
    #for pdb file check record
    done_pdb_handle = done_data_recorder(done_pdbs,str) 


    for c_num, c_line in enumerate(clusters):
        # to check if cluster already treated
        if done_file_handle.check_val_exist(c_num):
            continue    
        done_file_handle.add_val(c_num)  
        
        # to ignore headers in cluster file
        if c_line.startswith("#"):
            continue
        
        pdbs_in_same_cluster = c_line.split()
        
        # for one size cluster
        if len(pdbs_in_same_cluster) == 1:
            done_pdb_handle.add_val(pdbs_in_same_cluster[0])
            continue
     
        # first pdb for clustering
        number = 0
        while 1:
            if number >= len(pdbs_in_same_cluster):
                break
            
            if done_pdb_handle.check_val_exist(pdbs_in_same_cluster[number]+".pdb"):
                number += 1
                continue
                  
            
            if os.path.exists(path1+ pdbs_in_same_cluster[number]+".pdb"):     
                try:
                    pose_tmp = pose_from_pdb(path1+ pdbs_in_same_cluster[number]+".pdb")
                    break
                except RuntimeError:
                    if os.path.exists(path1+ pdbs_in_same_cluster[number]+".pdb"):
                        shutil.move(path1+ pdbs_in_same_cluster[number]+".pdb",path2+ pdbs_in_same_cluster[number]+".pdb" +".unread")
                    continue
                        

            number +=1
            
            
        
        if number >= len(pdbs_in_same_cluster):
            continue
        
        pose1 = pose_from_pdb(path1+ pdbs_in_same_cluster[number]+".pdb")     
        done_pdb_handle.add_val(pdbs_in_same_cluster[number]+".pdb")
        use_pdb_id = pdbs_in_same_cluster[number]+".pdb"
        for j in pdbs_in_same_cluster:
            j_pdb = j+".pdb"
            
            if done_pdb_handle.check_val_exist(j_pdb):
                continue
            
            if not os.path.exists(path1 +j_pdb):
               continue
            
            
            try:
                pose2 = pose_from_pdb(path1 + j_pdb)
            except RuntimeError:
                if os.path.exists(path1 +j_pdb):
                    shutil.move(path1 +j_pdb ,path2 +j_pdb +".unread")
                print(j_pdb + " can't be read.")
                continue
                
            pose2 = pose_from_pdb(path1 + j_pdb)
            done_pdb_handle.add_val(j_pdb)
            p1,c1 = protein_and_carb_size(pose1)
            p2,c2 = protein_and_carb_size(pose2)
            if  ((abs(p1-p2) < 10) & (c1 < c2)):
                pose1 = pose2.clone()
                if os.path.exists(path1 +use_pdb_id): 
                    shutil.move(path1 +use_pdb_id ,path2 +use_pdb_id)
                use_pdb_id = j_pdb
            else:
                if os.path.exists(path1 +j_pdb):           
                    shutil.move(path1 +j_pdb ,path2 +j_pdb)
                
            
        print(use_pdb_id)
            
        
        
        
    
if __name__ == '__main__':
    main()