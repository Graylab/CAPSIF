#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 15:51:18 2022
To make cluster file for glycan-interacting protein  names only.
works on directory containing glycan interaction proteins.


@author: sudhanshu
"""


def main():    
    # reading pdb cluster file
    file = "./data_files/clusters-by-entity-70.txt" # we used 70%identity
    fid = open(file,"r")
    data = fid.readlines()
    fid.close()

    # all readable pdbs names present in a dierctory
    all_pdbs = "/home/sudhanshu/HDD2/projects2/utils/glyco_data/all_pdbs_from_PDB_only/usable_pdb_for_motif_finding/2022_dataset_for_DL/pdb_selected_plus_lectins/readable_list.txt"
    fid = open(all_pdbs,"r")
    pdb_names = fid.readlines()
    fid.close()
    temp = []
    for i in pdb_names:
        temp.append(i.replace("\n",""))
    
    pdb_names = temp # all pdb names from directory
    del temp
    
    
    # clustering
    clusters =[]
    
    for i in data:
        split_i = i.split()
        pdbs_in_cluster = []
        chain_id = []
        for key in split_i:
            pdbs_in_cluster.append(key.split("_")[0])
            chain_id.append(key.split("_")[1])
        
        
        curr_clust =[]
        for j in pdb_names:
            if pdbs_in_cluster.count(j[:-4]) > 0:
                idx = pdbs_in_cluster.index(j[:-4])
                curr_clust.append(j[:-4]+"_"+chain_id[idx])
        
        if len(curr_clust) > 0:
            clusters.append(curr_clust)
    
    
    cluster_file = "./data_files/clusters.txt"    
    fid = open(cluster_file, "w+")
    fid.write("# Identified proteins + lectines with glycans clustered using rcsb sequence clusters 70% identity\n")
    for i in clusters:
        seq = ""
        for j in i:
            seq = seq + j +" "
        
        
        fid.write(seq[:-1]+"\n")
        
    fid.close()
    
    
    
if __name__ == "__main__":
    main()