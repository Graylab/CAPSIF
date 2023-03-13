#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 18:52:20 2022

@author: sudhanshu
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 14:05:48 2022

@author: sudhanshu
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from Bio.SeqUtils import seq3,seq1

from data_utils import aa_1_letter_code, aa_parameters_all, done_data_recorder

from pyrosetta import *
from pyrosetta.teaching import *
from pyrosetta.rosetta.core.pose import *
import random
import time

pdb_dir = "/Users/scanner1/Downloads/sid/data_set_5_may_2022/all_data/"
out_dir = pdb_dir + "/../backbone_sid/"

class pdb_to_interaction_file:
    def __init__(self,prot_file, out_dir = "./data_pdb_glycan_interaction_no_bound_all_nearby8A/",check_done=True, verbose = 1, use_glycan=1):

        self.verbose = verbose
        self.use_glycan = use_glycan
        if self.use_glycan ==1:  # for training
            init('-include_sugars  -ideal_sugars -ignore_unrecognized_res -renumber_pdb -ignore_zero_occupancy false  -alternate_3_letter_codes pdb_sugar -maintain_links -auto_detect_glycan_connections  -mute all')
        else: # can be used for prediction and non-readable glycans
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
        self.link_lists = []
        self.carb_res = []
        self.sasa_protein =[]
        self.sasa_radius = 1.4
        #self.protein_res_in_contact_with_carb = [] #starts with 1
        self.all_interacting_aa_from_all_atoms = [] #starts with 1
        self.randomize_times=0
        self.carb_interaction_cutoff= 4.2 #4.1 is less and 4.5 is very high
        self.save_data=1

    def run_me(self):

        if ((self.flag == True) or (self.carb_aa_distance_calc_and_save == 0)):
            if self.verbose == 1:
                print("currently treating:", self.pdb_file_name)
            self.init_variables()
            self.get_per_residue_sasa_for_protein_only()
            self.carb_all_interacting_residues_calculate()

            self.extract_CB_and_CA_xyz()
            self.all_parameters_for_residues()
            self.combine_data()
            #print(self.all_res_fixed_data)
            if self.save_data ==1:
                self.save_npz_pdb()
            else:
                return self.all_res_fixed_data, self.all_res_CB_CA_xyz

        else:
            return [],[-1]



    def get_per_residue_sasa_for_protein_only(self):
        '''
        This routine deletes not protein parts from the pose
        and calculates SASA per residue for the protein.
        '''
        temp_pose = self.pose_native.clone()

        dels =[] # to remove non protein residues
        for i,j in enumerate(temp_pose.sequence()):
            if j =='Z':
                if not temp_pose.residue(i+1).is_protein():
                    dels.append(i+1)

        dels.reverse()
        for i in dels:
            temp_pose.delete_residue_slow(i)

        # Per residue SASA calcul;atio step.
        rsd_sasa = pyrosetta.rosetta.utility.vector1_double()
        rsd_hydrophobic_sasa = pyrosetta.rosetta.utility.vector1_double()
        rosetta.core.scoring.calc_per_res_hydrophobic_sasa(temp_pose, rsd_sasa, rsd_hydrophobic_sasa, self.sasa_radius) #The last arguement is the probe radius
        del temp_pose
        self.sasa_protein = rsd_sasa


    def save_npz_pdb(self):
        '''
        Saves final minimalistic data required for training or testing
        file format is np archives.
        '''
        np.savez(self.out_dir + "/" + self.pdb_file_name + "_data.pdb.npz",
                all_res_fixed_data = self.all_res_fixed_data,
                all_res_CB_CA_xyz = self.all_res_CB_CA_xyz)



    def extract_CB_and_CA_xyz(self):
        '''
        Extracts CB and Ca coordinates from pdb file (not fixed)
        AND THE N AND C
        Also gets fixed data like residue properties for each residue

        '''
        pose = self.pose_use#pose_from_pdb(self.prot_file)
        all_res_fixed_data=[['Res#','Res','sasa','PDB_res_id','PDB_chain_ID','[1 and 0 for sugar type]','interact'],[]]
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
                xyz_N = np.array(pose.conformation().residue(res).xyz("N"))
                xyz_C = np.array(pose.conformation().residue(res).xyz("C"))

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
                all_res_CB_CA_xyz.append( xyz_N )
                all_res_CB_CA_xyz.append( xyz_CA )
                all_res_CB_CA_xyz.append( xyz_C )
                all_res_CB_CA_xyz.append( xyz_CB )

                pdb_res_number = pose.pdb_info().number(res)
                pdb_chain_number = pose.pdb_info().chain(res)
                pdb_phi = pose.phi(res)
                pdb_ome = pose.omega(res)
                pdb_psi = pose.psi(res)

                pdb_info.append( [pdb_res_number,pdb_chain_number,pdb_phi,pdb_ome,pdb_psi])


        #self.pose = pose
        self.only_protein_residues = only_protein_residues
        self.all_res_fixed_data = all_res_fixed_data
        self.all_res_CB_CA_xyz = np.array(all_res_CB_CA_xyz)
        self.pdb_info = pdb_info



    def all_parameters_for_residues(self):
        '''
        For each givine residue from pdb data, it calculates
        fixed property embedings ('hydr','arom','hdon', 'hacc','sasa')
        '''
        res_types = [seq3(self.aa_seq[i[1]-1]) for i in self.all_res_fixed_data[1] ]

        for i,aa_type in enumerate(res_types):
            params_for_curr_aa = self.aa_params[aa_type.upper()]
            aa_param_hydropathy = params_for_curr_aa[0]
            aa_param_aromaticity = params_for_curr_aa[2]
            aa_param_hbond_doner = params_for_curr_aa[3]
            aa_param_hbond_accept =params_for_curr_aa[4]
            sasa_v = self.sasa_protein[i+1]
            phi = self.pdb_info[i][2]
            psi = self.pdb_info[i][4]
            ome = self.pdb_info[i][3]

            self.all_res_fixed_data[1][i].append(aa_param_hydropathy)
            self.all_res_fixed_data[1][i].append(aa_param_aromaticity)
            self.all_res_fixed_data[1][i].append(aa_param_hbond_doner)
            self.all_res_fixed_data[1][i].append(aa_param_hbond_accept)
            self.all_res_fixed_data[1][i].append(sasa_v)
            self.all_res_fixed_data[1][i].append(phi)
            self.all_res_fixed_data[1][i].append(ome)
            self.all_res_fixed_data[1][i].append(psi)


    def carb_all_interacting_residues_calculate(self):
        '''
        Using Pyrosetta, this code identifies:
            1: carbohydrate residues
            2: all amino-acids interacting with carbohydrates (all atom).
            3: type of monosachrides and interacting residues

        '''
        pose = self.pose_native
        carb_res_pre = [i+1 for i,j in enumerate(pose.sequence()) if j =="Z" and pose.residue(i+1).is_carbohydrate()] # i+1 as residue starts from 1
        carb_res = []
        for i in carb_res_pre:
            if pose.glycan_tree_set().get_tree_root_of_glycan_residue(i) == 0:
                carb_res.append(i)


        self.carb_res = carb_res


        #print(self.carb_res_type)
        all_interacting_aa_from_all_atoms = []
        atom_interacting_with_type_of_sugar=dict()


        for i in carb_res:
            aa_interacting = self.all_iteracting_residues_from_a_residue(i)
            all_interacting_aa_from_all_atoms = np.concatenate((all_interacting_aa_from_all_atoms,aa_interacting), axis=0)

            res_name = pose.residue(i).carbohydrate_info().short_name()
            carb_res_id = self.monosaccharide_type_converter(res_name)

            if list(atom_interacting_with_type_of_sugar.keys()).count(carb_res_id)==0:
                atom_interacting_with_type_of_sugar[carb_res_id] = np.array([])

            atom_interacting_with_type_of_sugar[carb_res_id] =  np.concatenate(
                (atom_interacting_with_type_of_sugar[carb_res_id], aa_interacting), axis=0)

        for i in atom_interacting_with_type_of_sugar.keys():
            atom_interacting_with_type_of_sugar[i] = list(np.unique(
                atom_interacting_with_type_of_sugar[i]).astype(int))


        #print(atom_interacting_with_type_of_sugar)
        self.all_interacting_aa_from_all_atoms = list(np.unique(all_interacting_aa_from_all_atoms).astype(int))
        self.atom_interacting_with_type_of_sugar = atom_interacting_with_type_of_sugar



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

        nearest_CBs, sugar_coords = self.residues_with_nearest_cb_from_sugar_res( res_number)
        interacting_res = []

        for i in nearest_CBs:
            all_aa_atom_coords = self.all_atom_coordinates_non_H(i)

            for j in all_aa_atom_coords:
                for k in sugar_coords:
                    dist =  np.sqrt(np.sum((j-k)**2))

                    if dist < self.carb_interaction_cutoff:
                        interacting_res.append(i)
                        break


        return np.unique(interacting_res)




    def combine_data(self):

        for i in range(len(self.all_res_fixed_data[1])):

            # PDB RES ID
            self.all_res_fixed_data[1][i].append(self.pdb_info[i][0])

            # PDB CHAIN ID
            self.all_res_fixed_data[1][i].append(self.chains.index(
                self.pdb_info[i][1]))



            #CARB INT TYPE
            zeros_arr = [0,]* len(self.ms_list)
            for j in self.atom_interacting_with_type_of_sugar.keys():

                if self.atom_interacting_with_type_of_sugar[j].count(self.pdb_info[i][0]) > 0:
                    zeros_arr[j]=1

            # print(self.monosaccharide_type_converter(j),zeros_arr)

            for j in zeros_arr:
                self.all_res_fixed_data[1][i].append(j)





            # INTERACTION WITH CARB OR NOT
            if self.all_interacting_aa_from_all_atoms.count(
                    self.all_res_fixed_data[1][i][0]) > 0:
                self.all_res_fixed_data[1][i].append(1)
            else:
                self.all_res_fixed_data[1][i].append(0)






    def monosaccharide_type_converter(self, val):

        if type(val) == int:
            return self.ms_type[val]
        else:
            val = val.replace("-","")
            val = val+"-"
            for counter, i in enumerate(self.ms_list):
                if val.count(i+"-") > 0:
                    return counter


            return len(self.ms_list)-1

    def init_variables(self):
        self.chains = ' ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.ms_type = {    0:'Glcp',
                            1:'Galp',
                            2:'Manp',

                            3:'GlcpNAc',
                            4:'GalpNAc',

                            5:'Fucp',
                            6:'Xylp',

                            7:'GlcpA',
                            8:'GalpA',

                            9:'NeupAc',

                            10:'Others'}

        self.ms_list = list(self.ms_type.values())



if not os.path.exists(out_dir):
    os.mkdir(out_dir)


done_data = done_data_recorder("./done_pdbs_detailed.txt",str)


for fl in os.listdir(pdb_dir):
    if not fl.endswith('.pdb'):
        continue

    if done_data.check_val_exist(fl):
        continue
    done_data.add_val(fl)
    #print(fl)
    f = pdb_to_interaction_file(pdb_dir+fl, out_dir,0, verbose=0)
    f.carb_aa_distance_calc_and_save = 1
    f.run_me()
