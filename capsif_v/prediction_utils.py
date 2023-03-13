#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 17:35:46 2022

@author: sudhanshu
"""
#For prediction specific
import torch
import numpy as np
from utils import (
    load_checkpoint,
    dice,
)
import sys
sys.path.append("..")

from data_util import load_npz_data_mask_, load_npz_data_protein_
from settings import config1
from colorama import Fore, Style
import os

#set Model root Directory 
MOD_DIR = config1.model_root_directory


def load_model(model_type, sub_type=-1, DEVICE='cpu', dir=None): # 0 for 6, 1 for 26
    model_layers = [6,26,29,32][model_type]
    in_channel = model_layers#config2.model.in_channel
    out_channel = 1
    if in_channel == 6:
        print("depricated!")
        #depricated
    elif in_channel == 26:
        print("depricated!")
        #depricated
    elif in_channel == 29:
        
        from models import UNET_3D
        model_base = UNET_3D
        
        if sub_type == 0: #"CA"           
            check_point = MOD_DIR + config1.model_for_test_and_prediction_type0
            
        elif sub_type == 1: #CB    
            check_point = MOD_DIR + config1.model_for_test_and_prediction_type1
            
        elif sub_type == 2: # CACB_unit_vector  << ONLY PROVIDED
            check_point = MOD_DIR +config1.model_for_test_and_prediction_type2
            # check_point = MOD_DIR +"/models_DL/xxmy_checkpoint_best_36_2A_CACB_vector_coord_I_clean_data_rand05.pth.tar"
            
        elif sub_type == 3: # CACB_unit_vector  << ONLY PROVIDED
            check_point = MOD_DIR + config1.model_for_test_and_prediction_type3
            # check_point = MOD_DIR +"/models_DL/xxmy_checkpoint_best_36_2A_CACB_vector_coord_I_clean_data_rand05.pth.tar"
    if dir != None:
        check_point = dir;    
 
    print("Using",check_point)
    model = model_base(in_channels=in_channel, out_channels=out_channel).to(DEVICE)
    model.eval()
    load_checkpoint(torch.load(check_point,map_location=torch.device(DEVICE)), model,1 ,mode="test")  
    return model

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def nargout():
   import traceback
   callInfo = traceback.extract_stack()
   callLine = str(callInfo[-3].line)
   split_equal = callLine.split('=')
   split_comma = split_equal[0].split(',')
   return len(split_comma)

def predict_for_protein_and_mask(protein, real_mask, model, model_type, pose=None, save_npz=0):
    out_name = protein.split("/")[-1][:-4]
    out_save_folder = MOD_DIR +"/temp_files/saved_masks/"
    chimera_command =""
    if model_type == 1:
        x = torch.from_numpy(load_npz_data_protein_(protein, ))[:26,...].unsqueeze(0) #[:6,...]
    elif model_type == 2:
        x = torch.from_numpy(load_npz_data_protein_(protein, ))[:29,...].unsqueeze(0) #[:6,...]
    elif model_type == 3:
        x = torch.from_numpy(load_npz_data_protein_(protein, ))[:32,...].unsqueeze(0)
    else:
        x = torch.from_numpy(load_npz_data_protein_(protein, ))[:6,...].unsqueeze(0) #[:6,...]
    
    nrg = nargout()
    
    y = torch.from_numpy(load_npz_data_mask_(real_mask))
    aa_index = load_npz_data_mask_(protein,'layers')
    #x = x.to(device,dtype=torch.float)
    
    preds1 = model(x.float())    
    preds = (preds1 > 0.5)
    
    if save_npz == 1:
        np.savez(out_save_folder  + "/" + out_name +"_protein_1000_0.npz",
                layers = x[0],
                )       
        
        np.savez(out_save_folder + "/" + out_name + "_pred_1000_0.npz",
                  interaction= preds[0],
                  )
        
        np.savez(out_save_folder + "/" + out_name + "_real_1000_0.npz",
                  interaction= y,
                  )
        
    dice_score = dice(y,preds[0])
    
    if pose != None:
        residues = aa_index[torch.where(preds.squeeze(0) * aa_index> 0)].astype(int)
        true_residues = aa_index[torch.where(y == 1)].astype(int)
        
        if not type(true_residues) == np.ndarray:
            true_residues = np.array([true_residues])        
        if len(true_residues)> 1:
            true_residues.sort()
        
        
        if not type(residues) == np.ndarray:
            residues = np.array([residues])
        if len(residues)> 1:
            residues.sort()
        sent=''
        pdb_sent=''
        ground_truth = ''
        true_pos_str = ''
   
        only_p_array=[-1]
        for i in range(pose.size()):
            if pose.residue(i+1).is_protein():
                only_p_array.append(i+1)
        
        
        true_positive = intersection(residues, true_residues)
        
        #pdb_residues_predicted =[]
        #print(residues)
        for i in residues:
            sent = sent +str(i) +","
            pdb_sent = (pdb_sent + str(pose.pdb_info().number(only_p_array[i]))+"."
            + pose.pdb_info().chain(only_p_array[i]) +",")
            #pdb_residues_predicted.append(pose.pdb_info().number(only_p_array[i]))
            
        for i in true_residues:
            ground_truth = (ground_truth + str(pose.pdb_info().number(only_p_array[i]))+"."
            + pose.pdb_info().chain(only_p_array[i]) +",")
            
        for i in true_positive:
            true_pos_str = (true_pos_str + str(pose.pdb_info().number(only_p_array[i]))+"."
            + pose.pdb_info().chain(only_p_array[i]) +",")
           
        
        if nrg == 2:    
            print(Fore.RED + out_name +":" +Style.RESET_ALL )
            print(Fore.GREEN +"Dice score:" +Style.RESET_ALL, "%5.3f" % dice_score)
            print(Fore.GREEN + "Residues: "+Style.RESET_ALL+ sent[:-1] + " (Start from 1, only proteins)")    
            print(Fore.GREEN + "Residues: "+Style.RESET_ALL+ pdb_sent[:-1] + " (PDB numbering)")  
            print(Fore.GREEN + "Ground truth Residues: "+Style.RESET_ALL+ ground_truth[:-1] + " (PDB numbering)")  
            print(Fore.GREEN + "True positive Residues: "+Style.RESET_ALL+ true_pos_str[:-1] + " (PDB numbering)")  
            chimera_command = make_chimera_command(ground_truth, pdb_sent,true_pos_str)
            
            
        

            
    
    if nrg == 2:
        return dice_score, chimera_command
    
    return dice_score, chimera_command, [ground_truth, pdb_sent, true_pos_str]

def predict_for_protein_and_mask2(protein_vox, real_mask_vox, model, model_type, save_npz=0, DEVICE='cpuss'):
    
    chain_ids = ' ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    chimera_command =""
    if model_type == 1:
        x = protein_vox[:26,...].unsqueeze(0) #[:6,...]
    elif model_type == 2:
        x = protein_vox[:29,...].unsqueeze(0) #[:6,...]
    elif model_type == 3:
        x = protein_vox[:32,...].unsqueeze(0)
    else:
        x = protein_vox[:6,...].unsqueeze(0) #[:6,...]
    
    nrg = nargout()
    
    y = real_mask_vox.to(DEVICE)
    pdb_aa_index = protein_vox[-2,...].unsqueeze(0).numpy()
    pdb_chain_index = protein_vox[-1,...].unsqueeze(0).numpy()
    #x = x.to(device,dtype=torch.float)
    
    preds1 = model(x.float().to(DEVICE))    
    preds = (preds1 > 0.5)
    
    #print(preds1[preds])
    # print(preds[0])
    #x = x.to(device,dtype=torch.float)
            
    dice_score = dice(y,preds[0])
    
    #print(dice_score, preds[0].shape)
    predicted_index = torch.where(preds[0].squeeze(0).cpu() * pdb_aa_index> 0)    
    residues = pdb_aa_index[predicted_index].astype(int)
    chain = pdb_chain_index[predicted_index].astype(int)
    
    
    
    predicted_res_seq = np.stack((residues,chain)).transpose()
    #print(predicted_res_seq)
    if len(predicted_res_seq.shape) > 1:
        
        predicted_res_seq = predicted_res_seq[predicted_res_seq[:, 0].argsort()]
        predicted_res_seq = predicted_res_seq[predicted_res_seq[:, 1].argsort()]
    
    else:
        predicted_res_seq = np.stack(([residues],[chain])).transpose()
        
    predicted_seq = []
    for i,j in predicted_res_seq:
        predicted_seq.append(str(i)+"."+chain_ids[j])
        
     
    ground_index = torch.where(y.cpu() == 1)    
    true_residues = pdb_aa_index[ground_index].astype(int)
    true_chain = pdb_chain_index[ground_index].astype(int)
    
    ground_res_seq = np.stack((true_residues,true_chain)).transpose()
    
    
    if len(ground_res_seq.shape) > 1:
        ground_res_seq = ground_res_seq[ground_res_seq[:, 0].argsort()]
        ground_res_seq = ground_res_seq[ground_res_seq[:, 1].argsort()]
    else:
        ground_res_seq = np.stack(([true_residues],[true_chain])).transpose()
    
    ground_seq = []
    for i,j in ground_res_seq:
        ground_seq.append(str(i)+"."+chain_ids[j])
    
    true_positive = intersection(ground_seq, predicted_seq)
    
    #print( predicted_seq, ground_seq, true_positive )

    pdb_sent=''
    ground_truth = ''
    true_pos_str = ''
    
    for i in predicted_seq:
        pdb_sent = pdb_sent+ i +","
    
    for i in ground_seq:
        ground_truth = ground_truth + i + ","
        
    for i in true_positive:
        true_pos_str = true_pos_str + i +","
    
        
    if nrg == 2:    
        #print(Fore.RED + out_name +":" +Style.RESET_ALL )
        print(Fore.GREEN +"Dice score:" +Style.RESET_ALL, "%5.3f" % dice_score)        
        print(Fore.GREEN + "Residues: "+Style.RESET_ALL+ pdb_sent[:-1] + " (PDB numbering)")  
        print(Fore.GREEN + "Ground truth Residues: "+Style.RESET_ALL+ ground_truth[:-1] + " (PDB numbering)")  
        print(Fore.GREEN + "True positive Residues: "+Style.RESET_ALL+ true_pos_str[:-1] + " (PDB numbering)")  
        chimera_command = make_chimera_command(ground_truth, pdb_sent,true_pos_str)
    
    if nrg == 2:
        return dice_score, chimera_command
    
    return dice_score, chimera_command, [ground_truth, pdb_sent, true_pos_str]

def make_chimera_command(ground_truth, pdb_sent,true_pos_str):
    chimera_command =""
    if len(ground_truth[:-1]) > 0:
        chimera_command = chimera_command +"color red :" + ground_truth[:-1] + "; "
    if len(pdb_sent[:-1]) > 0:
        chimera_command = chimera_command +"color green :" + pdb_sent[:-1] + "; "
    if len(true_pos_str[:-1]) > 0:
        chimera_command = chimera_command + "color yellow :"+true_pos_str[:-1] +"; "
    
    chimera_command = chimera_command + "surface protein"
    print(Fore.GREEN + "For UCSF-Chimera: "+Style.RESET_ALL+ chimera_command)
    return chimera_command

def make_pymol_command(ground_truth, pdb_sent,true_pos_str):
    chimera_command =""

    l = ground_truth[:-1].split(',')
    if len(l) > 1:
        chimera_command += "sel gt, ";
        for i in range(len(l)):
            chimera_command += "(resi " + l[i][:-2] + " and chain " + l[i][-1] + ") or"
        #chimera_command = chimera_command +"color red :" + ground_truth[:-1] + "; "
    chimera_command = chimera_command[:-2] + "\n"

    l = pdb_sent[:-1].split(',')
    if len(l) > 1:
        chimera_command += "sel pred, ";
        for i in range(len(l)):
            chimera_command += "(resi " + l[i][:-2] + " and chain " + l[i][-1] + ") or"
        #chimera_command = chimera_command +"color red :" + ground_truth[:-1] + "; "
    chimera_command = chimera_command[:-2] + "\n"

    chimera_command += "show surface, all \n color gray80, all \n color red, gt \n color blue, pred \n color green, (gt and pred)\n"

    chimera_command += "set surface_quality, 1"


    #chimera_command = chimera_command + "surface protein"
    #print(Fore.GREEN + "For PyMol: "+Style.RESET_ALL+ chimera_command)
    return chimera_command

def command_run (command,current_settings):
    command = command.strip() 
    if len(command) < 1:
        print('Use "quit" to exit or "help" to list available commands.')
        return -2 #no action
    if command == 'ls':
        print([i for i in os.listdir("./") if i.endswith('pdb')])
        return 0
    if command == 'quit':
        return -1
    
    if command == 'use_chimera':
        current_settings.use_chimera = 1
        print("Chimera set to use!")
        return 0
    
    if command == 'stop_chimera':
        current_settings.use_chimera = 0
        print("Chimera stopped to use!")
        return 0

    if command == "load_glycan_on":
        current_settings.load_glycan = 1
        print("Glycan will be loaded from pdb!")
        return 0
     
    if command == "load_glycan_off":
        current_settings.load_glycan = 0
        print("Glycan will not be loaded from pdb!")
        return 0
    if command.startswith("crop_edge_by"):
        val= int(command.split(' ')[1])
        current_settings.crop_edge_value = val
        if val == 0:
            current_settings.cube_start_point = 1
        else:
            current_settings.cube_start_point = -1
            
        print("crop amount and random translation set to:", val)
        return 0

    if command == "clean_temp":
        #"For safety using delete file"
        if ( input("Are you sure to delete all files in temp(Y/N)?").upper() == "Y"):
            for fl in os.listdir(current_settings.data_dir):
                print(fl)
                if fl.endswith(".npz"):
                    os.remove(current_settings.data_dir +fl)
            
            for fl in os.listdir(current_settings.data_dir+"/cubes/"):
                if fl.endswith(".npz"):
                    os.remove(current_settings.data_dir+"/cubes/" +fl)
                    
            for fl in os.listdir(current_settings.data_dir+"/rcsb/"):
                if fl.count(".pdb")>0:
                    os.remove(current_settings.data_dir+"/rcsb/" +fl)
        return 0      
        

    if command == "help":
        print(current_settings.command_list)
        return 0
    
    if command.startswith("help"):
        print("Aurrently no description is added for the topic.")
        return -2

    if command.upper().startswith("RCSB:"):
        command = command.upper()
        if not os.path.exists(current_settings.data_dir):
            os.mkdir(current_settings.data_dir)
            
        if not os.path.exists(current_settings.data_dir+"rcsb/"):
            os.mkdir(current_settings.data_dir+"rcsb/")
        
        print("Downloading...")
        d=os.system("wget http://www.rcsb.org/pdb/files/"+command[5:]+".pdb -P "+ 
                    current_settings.data_dir+"rcsb/  -q")
        if d==0:
            command = current_settings.data_dir+"rcsb/" +command[5:]+".pdb"
        else:
            print("Cannot download!")
            command=0
        return command

    if not command.endswith("pdb"):
        command = command +".pdb"
        
    if not os.path.exists(command):
        print("file is not available.")
        return -1
    else:
        return command
    
    
    

#FOR PLOTTING
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def rsquare_data(d11,d22): #d11 : experminetal data X data
    
    nan_vals = d11+d22    
    d1 = d11[~np.isnan(nan_vals)]
    d2 = d22[~np.isnan(nan_vals)]    
    d1 = d1.reshape((-1,1))    
    model = LinearRegression()
    model.fit(d1, d2)
    r_sq = model.score(d1, d2)
    slope = model.coef_
    y_pred = model.predict(d1)  
    
    d1_min_pos = np.where(d1==min(d1))[0][0]
    d1_max_pos = np.where(d1==max(d1))[0][0]
    print(d1_max_pos)
    
    x_points = np.array([np.min(d1), np.max(d1)])
    y_points = np.array([y_pred[d1_min_pos], y_pred[d1_max_pos]])
    
    xy_out = [x_points, y_points]   
    
    
    return r_sq,slope,xy_out,~np.isnan(nan_vals)


def expression_compare2(d1,d2,xlb="xlb",ylb="ylb",fig_nm='fig_nm'):

    plus_dim = 0.1
    
     
    if (fig_nm != 'no_plot') :
        #score_match = ("%9.4f" % scrx[0]) + " | " +scrx[1] 
        #print("score:"+score_match )
        
        if (fig_nm != 'fig_nm'):
            fig=plt.figure()
            ax = fig.add_subplot(111)
        plt.title("SS")
        #plt.subplot(1,1,1,autoscale_on=True, aspect='equal',xlim=[0,1], ylim=[0,1])
        #plt.plot([min(d1),max(d1)],[min(d2),max(d2)],'--',linewidth=0.9,color="#D3D3D3");
                 
        d2_n = np.copy(d2)
        more_than_one = np.where(d2>1.0)[0]
        less_than_zero = np.where(d2<0.0)[0]
        if len(more_than_one)>0:
            d2_n[more_than_one] = 1
        if len(less_than_zero)>0:
            d2_n[less_than_zero] = 0
        
        
        plt.plot(d1,d2,'bo', markersize=4)
        
        r_sq,slope,y_pred, not_nan = rsquare_data(d1,d2)
        
        
        plt.plot(y_pred[0],y_pred[1],'-', color="darkred")
        
        plt.xlim([min(d1)-plus_dim,max(d1)+plus_dim])
        plt.ylim([min(d2)-plus_dim,max(d2)+plus_dim])  
        #plt.axis("equal")
        #print((max(d2)-min(d2))/(max(d1)-min(d1)))
        #plt.axes().set_aspect(aspect= ((max(d1)-min(d1))/(max(d2)-min(d2))))

        plt.xlabel(xlb,fontsize=9,labelpad=-1.)
        plt.ylabel(ylb,fontsize=9)
        plt.grid(1)
  
        #plt.title(r"$R^{2}:$" + ("%5.2f" % r_sq)+ "; Slope:"+ ("%5.2f" % slope), fontsize=8,pad=1.1)         
        plt.title(r"$R^{2}:$" + ("%5.2f" % (r_sq)) , fontsize=10,pad=1.1)         

        plt.tick_params(pad=1.1)        
        plt.xticks(fontsize=7.5, color="black",weight="bold")
        plt.yticks(fontsize=7.5, color="black",weight="bold")
        
        plt.axis("square")
        plt.xlim([0,1])
        plt.ylim([0,1])
       #print(i,j,counts)
    plt.tight_layout()
    if (fig_nm != 'fig_nm') and (fig_nm != 'no_plot') :
        fig.savefig(fig_nm, format='png', dpi=350)
    return [r_sq]
