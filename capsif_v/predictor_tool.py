#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 11:43:01 2022
last update 3/11/2023
@author: sudhanshu
"""

import sys
sys.path.append('..')
from prediction_utils import load_model, predict_for_protein_and_mask2, command_run
import time
import os
from data_preparation.pdb_2_interaction_file_converter import pdb_to_interaction_file
import subprocess
from colorama import Fore, Style
from utils import xyz_to_rotate_to_voxelize_to_translate, is_model_params_correct
#from utils import dice
import torch
import copy
import readline
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class dot_variable_class:
    def __init__(self):
        pass
    @property
    def all_methods(self):
        return list(self.__dict__.keys())
    
    
def main():
    
    if not is_model_params_correct():
        return
    
    # for arrow keybinding for previously used commands
    if os.path.exists("./.history_predict"):
        readline.read_history_file("./.history_predict")
    
    start_time = time.time()
        
    # all settings
    current_settings = dot_variable_class 
    current_settings.data_dir = sys.path[1] + "/temp_files/"
    current_settings.use_chimera = 1# if input("Do you want to use UCSF-CHIMERA to display the prediction (Y/N)?").upper() == 'Y'  else 0
    current_settings.command_list = ['ls','quit','use_chimera','stop_chimera', 'RCSB:XXXX', 'clean_temp','help','load_glycan_on', 'load_glycan_off','crop_edge_by']
    current_settings.load_glycan = 1
    current_settings.crop_edge_value = 0
    current_settings.cube_start_point=1
    
        
    multi_pdb = 1 # when to run server
    if len(sys.argv) > 1:
        multi_pdb = 0
        
    if multi_pdb == 1:
        commands_available = str(current_settings.command_list)[1:-1]
        print("Available commands:", Fore.RED + commands_available + Style.RESET_ALL)


    # See load model for these values
    select_model = 2 # default is 2 i.e. UNET-3D2 and 29layers
    models = [[2,0], #29 #CA 0
              [2,1], #29 #CB 1
              [2,2], #29 #vector CB CA 2
              [2,3],] #29 #UNETRESNET vector CB CA 3
    
     
    use_layers = [6,26,29,32]  

    print("Loading model...")
    model = load_model(models[select_model][0],models[select_model][1], DEVICE=DEVICE )
    model_time = time.time()
    print("Model load time: ","%5.1f " % (model_time -start_time), "seconds.\n")
    

    # intiating pdb to npz converter file reader
    pdb_npz_file_reader = xyz_to_rotate_to_voxelize_to_translate()
    # pdb_npz_file_reader.return_res_numbers_from_1 = 1
    pdb_npz_file_reader.max_rotation_plus_minus = 0
    pdb_npz_file_reader.max_pixel_translate_per_axis = 0
    pdb_npz_file_reader.use_res_index = 1
    pdb_npz_file_reader.layers_use = use_layers[ models[select_model][0]  ]
    pdb_npz_file_reader.layer_29_type = models[select_model][1]  
    pdb_npz_file_reader.cube_start_points = current_settings.cube_start_point
    
    #Prediction block
    while 1:   
        
        start_time = time.time()   
        
        if multi_pdb == 0:
            pdb_file1 = sys.argv[1]
        else:
            pdb_file1 = input(Fore.GREEN + "input PDB file:" +Style.RESET_ALL)
            readline.write_history_file("./.history_predict")
            
        pdb_file2 = copy.copy(pdb_file1)
            
        # command run is main part to handle pdbs and other commands
        return_val = command_run(pdb_file2, current_settings)       
        if return_val == 0: # for setting different variables
            continue
        elif return_val == -1: # for exiting
            break
        elif return_val == -2: # for no action
            continue
        else:
            pdb_file = return_val # for prediction
        
        pdb_file_name_wo_path = pdb_file.split("/")[-1][:-4]
        
        # to make temporary NPZ file from protein
        if not os.path.exists(current_settings.data_dir):
            os.mkdir(current_settings.data_dir)            
        out_file = current_settings.data_dir+pdb_file_name_wo_path +".npz"
        
        if os.path.exists(out_file):
            os.remove(out_file)
            
        #           
        print("Loading and Voxelizing Coordinates")
        f = pdb_to_interaction_file( pdb_file, current_settings.data_dir,
                                    0, verbose=0, 
                                    use_glycan=current_settings.load_glycan)
        
        f.carb_aa_distance_calc_and_save = 1   
        f.save_data =0
        s,x = f.run_me() 
        
        # Error handelings
        if len(x) <= 1:
            
            if len(x) == 0:
                print("Can not read pdb file! Not a protein data.")
                continue
            if x[0] == -1:
                print("PyRosetta could not read glycan!")
                print("Check PDB file (ring glycan, clashes etc.)!")
                print("or use restart the code and run 'load_glycan_off' flag. Dice will be 0")
                continue
        
        # for special cases to remove edge effect
        pdb_npz_file_reader.crop_extra_edge = current_settings.crop_edge_value
        pdb_npz_file_reader.cube_start_points = current_settings.cube_start_point
        
        #pdb data to voxels with ground truth 
        proteins,masks = pdb_npz_file_reader.apply(x,s)
        
        np.savez(out_file + ".npz",
                layers = proteins)
        
        np.savez(out_file[:-4] + "_mask.npz",
                  interaction= masks
                  )
        

        print("Total data handling time: ","%5.1f " % (time.time() -start_time), "seconds.")
       
        start_time = time.time()
        print("predicting now ..")   
        d,cmd=predict_for_protein_and_mask2(torch.from_numpy(proteins), torch.from_numpy(masks), model,  models[select_model][0] ,save_npz=0, DEVICE=DEVICE)
        #d,cmd=predict_for_protein_and_mask(proteins, masks, model, model_type,f.pose_native,save_npz=0)

        pred_time = time.time()
        #print("Prediction time: ","%5.1f " % (pred_time -cube_time), "seconds.")
        print("Total prediction time: ","%5.1f " % (pred_time -start_time), "seconds.")
        
       
        # Sending output to UCSF CHIMERA. 
        # CHIMERA should be accesible by 'chimera' command
        if current_settings.use_chimera == 1:
            fid =open("/tmp/chimera_command.cmd","w+")   
            
            tmp = subprocess.Popen(['realpath', pdb_file ],stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            pdb_file_with_path,err = tmp.communicate()
            
            fid.write("close all\n")
            fid.write("background solid white\n")
            fid.write("open "+pdb_file_with_path.decode('utf-8').strip()+"\n")
            fid.write("reset\n")
            fid.write("set depthCue\n")
            fid.write(cmd+"\n")
            fid.close()
            
            os.system("chimera --send /tmp/chimera_command.cmd")
        
        
        if multi_pdb == 0:
            break
    
    readline.write_history_file("./.history_predict")
    
    
    
if __name__ == '__main__':
    main()