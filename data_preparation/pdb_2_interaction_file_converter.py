#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 14:05:48 2022

@author: sudhanshu
"""
import numpy as np
import matplotlib.pyplot as plt

import os
from Bio.SeqUtils import seq3,seq1

from data_preparation.data_utils import aa_1_letter_code, aa_parameters_all, done_data_recorder

from pyrosetta import *
from pyrosetta.teaching import *
from pyrosetta.rosetta.core.pose import *
import random
import time
import sys
# sys.path.append("..")
# from settings import config1


class pdb_to_interaction_file:
    def __init__(self,prot_file, out_dir = "./data_pdb_glycan_interaction_no_bound_all_nearby8A/",check_done=True, verbose = 1, use_glycan=1):

        self.verbose = verbose
        self.use_glycan = use_glycan
        if self.use_glycan ==1:
            init('-include_sugars  -ideal_sugars -ignore_unrecognized_res -renumber_pdb -ignore_zero_occupancy false  -alternate_3_letter_codes pdb_sugar -maintain_links -auto_detect_glycan_connections  -mute all')
        else:
            init('-ignore_unrecognized_res -ignore_zero_occupancy false  -mute all')

        self.prot_file = prot_file
        self.pdb_file_name = prot_file.split("/")[-1][:-4]
        if self.verbose == 1:
            print("Starting pdb file:", self.pdb_file_name)
        self.aa_distance_map = 0
        self._done_count_ = 1
        self.carb_aa_distance_calc_and_save = 0
        self.out_dir = out_dir
        self.aa_params = aa_parameters_all()
        self.use_trial_surface = 0
        try:
            self.pose_native = pose_from_pdb(prot_file)
            self.pose_use = self.pose_native.clone()
            self.flag = True
            #self.link_lists.append("PDB: " + self.pdb_file_name)
        except RuntimeError:
            if self.verbose == 1:
                print("PDB: " + self.pdb_file_name + ": Can't read!")
            self.flag = False

        if check_done == True:
            self.check_done_pdb_list()

        self.aa_seq = aa_1_letter_code()
        self.cube_size_A = 36
        self.voxel_size_A = 2
        self.link_lists = []
        self.carb_res = []
        self.carb_pdb = []
        self.sasa_protein =[]
        self.sasa_radius = 1.4
        #self.protein_res_in_contact_with_carb = [] #starts with 1
        self.all_interacting_aa_from_all_atoms = [] #starts with 1
        self.randomize_times=0
        self.carb_interaction_cutoff= 4.2 #4.1 is less and 4.5 is very high
        self.save_data=1


        if ((self.flag == True) or (self.carb_aa_distance_calc_and_save == 0)):
            pass
            #self.all_residue_distances()
            #self.carb_aa_interaction_calculate()
            #self.carb_COM_aa_interaction_calculate()


    def run_me(self):

        if ((self.flag == True) or (self.carb_aa_distance_calc_and_save == 0)):
            if self.verbose == 1:
                print("currently treating:", self.pdb_file_name)
            self.get_per_residue_sasa_for_protein_only()
            self.carb_all_interacting_residues_calculate()

            self.extract_CB_and_CA_xyz()
            self.all_parameters_for_residues()
            self.combine_data()
            if self.save_data ==1:
                self.save_npz_pdb()
            else:
                return self.all_res_fixed_data, self.all_res_CB_CA_xyz


            # self.all_residue_distances2()
            # #self.carb_aa_interaction_calculate()
            # self.carb_COM_aa_CB_interaction_calculate()

            # self.all_residue_distances2('CA','CA')
            # self.carb_COM_aa_CB_interaction_calculate()
        else:
            return [],[-1]

    def get_per_residue_sasa_for_protein_only(self):
        temp_pose = self.pose_native.clone()

        dels =[]
        for i,j in enumerate(temp_pose.sequence()):
            if j =='Z':
                if not temp_pose.residue(i+1).is_protein():
                    dels.append(i+1)


        dels.reverse()
        for i in dels:
            temp_pose.delete_residue_slow(i)
        rsd_sasa = pyrosetta.rosetta.utility.vector1_double()
        rsd_hydrophobic_sasa = pyrosetta.rosetta.utility.vector1_double()
        rosetta.core.scoring.calc_per_res_hydrophobic_sasa(temp_pose, rsd_sasa, rsd_hydrophobic_sasa, self.sasa_radius) #The last arguement is the probe radius
        del temp_pose
        self.sasa_protein = rsd_sasa


    def save_npz_pdb(self):
        np.savez(self.out_dir + "/" + self.pdb_file_name + "_data.pdb.npz",
                all_res_fixed_data = self.all_res_fixed_data,
                all_res_CB_CA_xyz = self.all_res_CB_CA_xyz)



    def extract_CB_and_CA_xyz(self):
        pose = self.pose_use#pose_from_pdb(self.prot_file)
        all_res_fixed_data=[['Res#','Res','hydr','arom','hdon', 'hacc','sasa','PDB_res_id','PDB_chain_ID','interact'],[]]
        all_res_CB_CA_xyz = []
        only_protein_residues=[-1]  # for zero index
        pdb_info = []
        for res in range(1, pose.size()+1):
            if pose.conformation().residue(res).is_protein():
                only_protein_residues.append(res)
                CB = 'CB'
                if pose.conformation().residue(res).name1() == "G":
                    CB = "CA"


                xyz_CB = np.array(pose.conformation().residue(res).xyz(CB))
                xyz_CA = np.array(pose.conformation().residue(res).xyz('CA'))

                res1_name = pose.conformation().residue(res).name1()
                res3_name = pose.conformation().residue(res).name3()

                if res3_name.upper() =='GLX':
                    print('Glx is treated as Gln!')
                    res1_name ='Q'
                    res3_name ='Gln'

                if ['CSO','CSD'].count( res3_name.upper()) > 0 :
                    print(res3_name + " is treated as Cys!")
                    res1_name ='C'
                    res3_name ='Cys'

                if res1_name == 'Z':
                    print("Unknown residue ", res3_name, " found!")
                    print("Ignoring data collection for residue!")
                    continue


                input_mat = [ res,self.aa_seq.index(res1_name)+1 ]
                all_res_fixed_data[1].append( input_mat )  # to identify carb-interactions map
                all_res_CB_CA_xyz.append( xyz_CB )
                all_res_CB_CA_xyz.append( xyz_CA )



                pdb_res_number = pose.pdb_info().number(res)
                pdb_chain_number = pose.pdb_info().chain(res)
                pdb_info.append( [pdb_res_number,pdb_chain_number])

        #self.pose = pose
        self.only_protein_residues = only_protein_residues
        self.all_res_fixed_data = all_res_fixed_data
        self.all_res_CB_CA_xyz = np.array(all_res_CB_CA_xyz)
        self.pdb_info = pdb_info

    def all_parameters_for_residues(self):
        res_types = [seq3(self.aa_seq[i[1]-1]) for i in self.all_res_fixed_data[1] ]

        for i,aa_type in enumerate(res_types):
            params_for_curr_aa = self.aa_params[aa_type.upper()]
            aa_param_hydropathy = params_for_curr_aa[0]
            aa_param_aromaticity = params_for_curr_aa[2]
            aa_param_hbond_doner = params_for_curr_aa[3]
            aa_param_hbond_accept =params_for_curr_aa[4]
            sasa_v = self.sasa_protein[i+1]

            self.all_res_fixed_data[1][i].append(aa_param_hydropathy)
            self.all_res_fixed_data[1][i].append(aa_param_aromaticity)
            self.all_res_fixed_data[1][i].append(aa_param_hbond_doner)
            self.all_res_fixed_data[1][i].append(aa_param_hbond_accept)
            self.all_res_fixed_data[1][i].append(sasa_v)


    def carb_all_interacting_residues_calculate(self):
        pose = self.pose_native
        carb_res_pre = [i+1 for i,j in enumerate(pose.sequence()) if j =="Z" and pose.residue(i+1).is_carbohydrate()] # i+1 as residue starts from 1
        carb_res = []
        for i in carb_res_pre:
            if pose.glycan_tree_set().get_tree_root_of_glycan_residue(i) == 0:
                carb_res.append(i)


        self.carb_res = carb_res
        all_interacting_aa_from_all_atoms = []
        for i in carb_res:
            aa_interacting = self.all_iteracting_residues_from_a_residue(i)


            all_interacting_aa_from_all_atoms = np.concatenate((all_interacting_aa_from_all_atoms,aa_interacting), axis=0)

        self.all_interacting_aa_from_all_atoms = list(np.unique(all_interacting_aa_from_all_atoms).astype(int))
        self.carb_to_pdb()

    def carb_to_pdb(self):
        self.carb_pdb = [];
        for i in self.carb_res:
            self.carb_pdb.append(self.pose_native.pdb_info().pose2pdb(i))

    def all_atom_coordinates_non_H(self, res_number):
        pose = self.pose_native
        num_of_atoms = pose.residue(res_number).natoms()
        all_atom_coordinates = []
        for i in range(num_of_atoms):
            atom_name = pose.residue(res_number).atom_name(i+1).strip()
            if atom_name.count('H')> 0:
                continue
            if atom_name.startswith('V')> 0:
                continue

            all_atom_coordinates.append( np.array(pose.residue(res_number).atom(i+1).xyz()))
        return all_atom_coordinates


    def residues_with_nearest_cb_from_sugar_res(self, res_number, cutoff = 10):
        pose = self.pose_native
        all_sugar_coords = self.all_atom_coordinates_non_H(res_number)
        nearest_CBs = []


        for i in range(pose.size()):
            CB='CB'
            if not pose.residue(i+1).is_protein():
                continue
            if pose.residue(i+1).name1() == "G":
                CB = 'CA'


            xyz_CB = np.array(pose.residue(i+1).xyz(CB))
            for j in all_sugar_coords:
                dist =  np.sqrt(np.sum((j-xyz_CB)**2))

                if dist < cutoff:
                    nearest_CBs.append(i+1)
                    break


        nearest_CBs.sort()
        nearest_CBs = np.unique(nearest_CBs)
        return nearest_CBs, all_sugar_coords


    def all_iteracting_residues_from_a_residue(self, res_number):
        cutoff = self.carb_interaction_cutoff
        nearest_CBs, sugar_coords = self.residues_with_nearest_cb_from_sugar_res( res_number)
        interacting_res = []

        for i in nearest_CBs:
            all_aa_atom_coords = self.all_atom_coordinates_non_H(i)

            for j in all_aa_atom_coords:
                for k in sugar_coords:
                    dist =  np.sqrt(np.sum((j-k)**2))

                    if dist < cutoff:
                        interacting_res.append(i)
                        break


        return np.unique(interacting_res)

    def combine_data(self):
        chains = ' ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for i in range(len(self.all_res_fixed_data[1])):



            self.all_res_fixed_data[1][i].append(self.pdb_info[i][0])
            self.all_res_fixed_data[1][i].append(chains.index(self.pdb_info[i][1]))

            if self.all_interacting_aa_from_all_atoms.count(self.all_res_fixed_data[1][i][0]) > 0:
                self.all_res_fixed_data[1][i].append(1)
            else:
                self.all_res_fixed_data[1][i].append(0)

    
if __name__ == "__main__":
    
    pdb_dir = config1.dataset_dir + "/test_af/"
    out_dir = pdb_dir + "pdb_npz/"
            
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    
    done_data = done_data_recorder(pdb_dir+"done_pdbs.txt",str)    
    
    for fl in os.listdir(pdb_dir):
        if not fl.endswith('.pdb'):
            continue
        
        if done_data.check_val_exist(fl):
            continue
        done_data.add_val(fl)
        
        f = pdb_to_interaction_file(pdb_dir+fl, out_dir,0, verbose=0)
        f.carb_aa_distance_calc_and_save = 1    
        f.run_me()
        
    