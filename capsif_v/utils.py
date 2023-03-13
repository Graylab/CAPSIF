#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 15:34:54 2022
Last update: 3/4/2023

@author: sudhanshu
"""
# from importlib.abc import Loader
# from isort import file
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import sys

sys.path.append("..")
from data_preparation.data_utils import aa_1_letter_code
from data_util import ThreeDDataset,PDB_NPZ_Dataset
from settings import config1
# For training specific

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    print("=> Saved checkpoint")
    


def load_checkpoint(checkpoint, model, optimizer, mode = "train"):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"]) 
    if mode == "train":
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["epoch"], checkpoint["best"]


def split_indices(n,val_part):
    n_val = int(val_part*n)
    idxn = np.random.permutation(n)
    return idxn[n_val:],idxn[:n_val]

def len_of_data(directory_,ext=".npz"):
    count = 0
    for i in os.listdir(directory_):
        if i.endswith(ext):
            count += 1
    return count


def get_loaders( train_dir, val_dir, batch_size, data_reader_and_transform,
    num_workers=4, pin_memory=True, layers = 0 ):
    
    train_ds = PDB_NPZ_Dataset( protein_dir=train_dir, transformer=data_reader_and_transform, train=1)    
    train_loader = DataLoader( train_ds, batch_size=batch_size, num_workers=num_workers,
        pin_memory=pin_memory, shuffle=True )

    val_ds = PDB_NPZ_Dataset( protein_dir=val_dir, transformer=data_reader_and_transform)
    val_loader = DataLoader( val_ds, batch_size=1, num_workers=num_workers,
        pin_memory=pin_memory, shuffle=False )

    return train_loader, val_loader



def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()


    with torch.no_grad():
        for x, y in loader:
            x = x.to(device,dtype=torch.float)
            y = y.to(device, dtype=torch.float).unsqueeze(1)
            #print(x.shape)
            #preds = torch.sigmoid(model(x))
            preds = model(x)
            #print(preds)
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2*(preds * y).sum()) /((preds +y).sum() + 1e-8)

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )

    print (f"Dice score: {dice_score/len(loader)}")
    model.train()


def calc_accuracy(loader, model, device="cuda"):
    dice_score = 0
    model.eval()
    with torch.no_grad():
        for count, (x, y) in enumerate(loader):
            x = x.to(device,dtype=torch.float)
            y = y.to(device, dtype=torch.float).unsqueeze(1)

            #preds = torch.sigmoid(model(x))
            preds = model(x)
            #print(preds)
            preds = (preds > 0.5).float()
            dice_score +=dice(y,preds)
            
    model.train()
    return dice_score/count


def save_predictions_as_masks(loader, model, folder= "saved_masks", device ="cuda"):
    model.eval()

    for idx, (x,y) in enumerate(loader):        
        x = x.to(device=device, dtype=torch.float)
        with torch.no_grad():
            #preds = torch.sigmoid(model(x))
            preds = model(x)
            preds = (preds > 0.5).float()

        xx = x.cpu()
        for idx2, i in enumerate(range(xx.shape[0])):
            np.savez(folder + "/" + f"protein_{idx}_{idx2}.npz",
                    layers = xx[i,...],
                    )
    
    
            np.savez(folder + "/" + f"pred_{idx}_{idx2}.npz",
                     interaction= preds[i].cpu(),
                     )
    
            np.savez(folder + "/" + f"real_{idx}_{idx2}.npz",
                     interaction= y[i].cpu(),
                     )

    model.train() 
    
    
def dice(y_true, y_pred, smoothing_factor=0.01):
    y_true_f = torch.flatten(y_true,1)
    
    y_pred_f = torch.flatten(y_pred,1)
    # print(y_true_f, y_pred_f,"----")
    intersection = torch.sum(y_true_f * y_pred_f)
    return ((2. * intersection + smoothing_factor)
            / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smoothing_factor))


def dice_loss(y_true, y_pred):
    #print(y_true.shape, y_pred.shape)
    return -dice(y_true, y_pred)

    
def report_file_name(directory, use_previous=0):
    file_init = directory + "/report_"
    for i in range(1000):
        trial_file = file_init + ("%4d" % (i+1)).replace(" ","0") + ".dat"
        if not os.path.exists(trial_file):
            if use_previous == 1:
                trial_file = file_init + ("%4d" % (i)).replace(" ","0") + ".dat"
            break
        
    return trial_file
        
    
    
class xyz_to_rotate_to_voxelize_to_translate:
    # if the size is bigger than cube_size
    #it randomely cuts data to fit in a cube
    
    def __init__(self):
        self.voxel_size_A = 2
        self.cube_size = 36
        self.aa_seq = aa_1_letter_code()
        self.max_pixel_translate_per_axis = 3
        self.max_rotation_plus_minus = 180
        self.cube_start_points = -1 # -1 for random
        self.use_CA_vetor = 1
        self.use_res_index = 0 # for prediction output in predictor
        self.return_res_numbers_from_1 = 0
        self.layers_use = 32
        self.layer_29_type = 0 #CA=0 #CB=1 #CB-CArray=3 works for layers_use=29
        self.crop_extra_edge = 0
        self.randomize_by = 0.0 #A # for training only
        #self.pdb_f=pdb_functions() only for debugging
        

    def apply(self, xyz_data, static_data, train =0):   
        self.xyz_data = xyz_data
        self.rotate_coordinates_by( self.max_rotation_plus_minus )
        
        self.xyz_beta_data = self.xyz_data[range(0,len(self.xyz_data),2),:]
        self.xyz_alpha_data = self.xyz_data[range(1,len(self.xyz_data),2),:]
        beta_rand = 0
        alpha_rand = 0
        # print("HI", self.randomize_by, train)
        if train == 1:
            if self.randomize_by > 0:
                beta_rand = (np.random.rand(self.xyz_beta_data.shape[0], 
                                            self.xyz_beta_data.shape[1])*2 -1)*self.randomize_by
                alpha_rand = (np.random.rand(self.xyz_alpha_data.shape[0],
                                             self.xyz_alpha_data.shape[1])*2 -1)*self.randomize_by
        
        self.xyz_beta_data = self.xyz_beta_data + beta_rand
        self.xyz_alpha_data = self.xyz_alpha_data + alpha_rand
        
     
        self.static_data = static_data        
        # Send  translational details to next routine  
        self.change_coordinates_for_voxel_dimension(self.xyz_beta_data)#, translate_data)
        self.calculate_unit_vectors()
        self.voxelize()
        return self.voxel_layers, self.voxel_mask
        
 
    def change_coordinates_for_voxel_dimension(self, xyz_coords):
        n = self.voxel_size_A   
        
        xyz_vox = np.round( xyz_coords/n)
        min_ = np.min(xyz_vox, 0)
        xyz_vox = (xyz_vox - min_).astype(int)  # normalizing    
        
        max_vals = np.max( [ np.max(xyz_vox, 0) - self.cube_size +1 ,
                            [0,0,0]],0) +1 + int(self.crop_extra_edge)
        
        
        away_from_center = (np.max([self.cube_size-np.max(xyz_vox,0),[0,0,0]],0)/2).astype(int)-1# finds movable pixels
        
        away_from_center = np.max([away_from_center,np.ones(3)*self.crop_extra_edge/2],0)
        
        # print(away_from_center)
        counter_ijk =np.zeros(3)
        counter = 0
        while 1: # to solve emply voxel problems
            if counter <=50:
                cube_start_points = np.zeros(3)+self.cube_start_points + counter_ijk 
            if self.cube_start_points == -1:
                cube_start_points = np.random.randint(max_vals) 
                
            
        
            translate_amount = np.zeros(3)
            if self.max_pixel_translate_per_axis != 0:
                translate_amount = (np.min(
                    [ np.random.randint((self.max_pixel_translate_per_axis,)*3), 
                    away_from_center],0) *  
                    ((np.random.randint((2,)*3) -0.5)/0.5)).astype(int)
                
             
            
            translate_from_center = (translate_amount-away_from_center).astype(int) 
            
            
            # print(translate_amount,away_from_center,translate_amount-away_from_center)
            # print(max_vals)
            # print(cube_start_points)
             
            # print(cube_start_points,max_vals)
            xyz_use_index1 = xyz_vox >= cube_start_points
            xyz_use_index2 = xyz_vox < (cube_start_points + self.cube_size - self.crop_extra_edge)
            
            indexes = xyz_use_index1 & xyz_use_index2
            indexes = np.sum(indexes,1)==3
            #print(counter_ijk, max_vals, counter ,sum(indexes), len(indexes))
            if sum(indexes) > 5: # minimum coordinates should be 5
                break
                  
            for pit in range(3):
                if counter_ijk[pit] <= counter:
                    counter_ijk[pit] += 1    
                    if sum(counter_ijk == counter_ijk[0])== 3:
                        counter +=1
           
                    break
                
            if counter > 50: #HARD measure  
                #print ("index trap!")       
                counter_ijk =np.zeros(3)
                cube_start_points = np.random.randint(max_vals) 
            
            
        xyz_vox = xyz_vox[indexes,:]
        
        # if ( (xyz_vox[:,0].size == 0) or (xyz_vox[:,1].size == 0) or (xyz_vox[:,1].size == 0)):
        #   
        #print(xyz_vox.shape)
        try:
            xyz_vox[:,0] -= min(xyz_vox[:,0]) + translate_from_center[0]
            xyz_vox[:,1] -= min(xyz_vox[:,1]) + translate_from_center[1]
            xyz_vox[:,2] -= min(xyz_vox[:,2]) + translate_from_center[2]
        except:
            
            print("hey" ,xyz_vox.shape)
            return 0
            
            
        self.xyz_vox = xyz_vox        
        self.indexes = indexes
        
    def rotation_mat(self,theta,axis):
            pi=np.pi
            theta=theta*pi/180
            a = np.cos(theta/2)
            b,c,d = -axis*np.sin(theta/2)
            rot_mat=np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                             [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                             [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])                             
             
            return (rot_mat)#+origin)   

        
    def rotate_coordinates_by(self,alfa):
        
        alfa = ((np.random.rand(1)*2)-1)*alfa
        if not alfa == 0:
            rotation_xyz = self.xyz_data
            origin = np.mean(self.xyz_data,0) # center of coordinates
            axist= rotation_xyz[np.random.randint(len(self.xyz_data))] - origin # random axis from CB
    
            axist = axist/np.sqrt(np.dot(axist,axist))
            rot_m = np.squeeze(self.rotation_mat(alfa,axist))
            t_data = rotation_xyz-origin
            
            for j in range (self.xyz_data.shape[0]):
                c_data=t_data[j]
                t_data[j]=np.dot(rot_m,c_data)          
            
            self.xyz_data = t_data +origin
        
    # Center yaa translate rotate        
        
    def calculate_unit_vectors(self):        
        ## Calculating unit vector for required data only
        unit_v = self.xyz_alpha_data[self.indexes,:] - self.xyz_beta_data[self.indexes,:]
        # for i in unit_v:
        #     print (i)
               
        self.use_CA_for_indexes = self.xyz_alpha_data[self.indexes,:]
        self.use_CB_for_indexes = self.xyz_beta_data[self.indexes,:]
        
        dist_v = np.sqrt(np.sum( unit_v**2,1))
        for i in range(len(unit_v)):
            if sum(unit_v[i]==0) == 3 :
                continue
            unit_v[i] = unit_v[i] / dist_v[i]
            
        self.unit_v = unit_v
        
    def voxelize(self):
        self.usable_static_data = np.array(self.static_data[1])[self.indexes]
        self.voxel_mask = []
        self.voxel_layers = []
        
        
        ca_cb_layers = 6
        if self.layers_use == 29:
            ca_cb_layers =3
            
        res_ind_layer = 0
        if self.use_res_index == 1:
            res_ind_layer = 2  # one for res number, one for chain
            
        
        property_layers = 5 # +1 for debuguibg case
        total_layers =   (1+               #Voxel                          
                         property_layers+  #PROPERTIES
                         20+               #ONE-HOT-AA
                         ca_cb_layers +               #for CA vector
                         res_ind_layer)
                         # 1+                # AA number for only recovery
                         # 1                 # interaction  # not used in training etc. 
                         # )
        
     
        layers = np.zeros((total_layers,)+(self.cube_size,)*3)
        mask = np.zeros((1,)+(self.cube_size,)*3)
        
        if self.return_res_numbers_from_1 == 1:
            res_mat_from_1 = np.zeros((1,)+(self.cube_size,)*3)
        # print(layers.shape, layers[0].shape,)
        # print("here",np.max(self.xyz_vox,0),"done")
        
    
        
        for counter, (i, j, k) in enumerate(self.xyz_vox):            
            static_data_for_res = self.usable_static_data[counter]
            # print(i,j,k)
            aa_param_hydropathy = static_data_for_res[2]
            aa_param_aromaticity = static_data_for_res[3]
            aa_param_hbond_doner = static_data_for_res[4]
            aa_param_hbond_accept = static_data_for_res[5]
            
            aa_sasa = static_data_for_res[6] # index start from 1
            # if ((i>34) or (i>34) or (i>34)):
            #     print(i,j,k)
            
            layers[0,i,j,k] = 1  # Voxel Layer
            layers[1,i,j,k] = aa_param_hydropathy  # hydrophobic_layer
            layers[2,i,j,k] = aa_param_aromaticity  # aromatic_layer
            layers[3,i,j,k] = aa_param_hbond_doner
            layers[4,i,j,k] = aa_param_hbond_accept                        
            layers[5,i,j,k] = aa_sasa  # SASA
            #layers[4,i,j,k] = counter  # for debugging
            
            # One hot layers for amino-acids
            #print(aa_type)
            aa_number = int(static_data_for_res[1] -1)
           
            aa_index = 1 + property_layers  + aa_number # after property layers
            layers[aa_index,i,j,k] = 1

            # CA unit vectors   26 27 28        X Y Z
            
            # layers[26,i,j,k] = self.unit_v[counter][0]*self.use_CA_vetor
            # layers[27,i,j,k] = self.unit_v[counter][1]*self.use_CA_vetor
            # layers[28,i,j,k] = self.unit_v[counter][2]*self.use_CA_vetor
            
            #Trying CB and CA coordinate3s
            if self.layers_use == 32:
                layers[26,i,j,k] = self.use_CA_for_indexes[counter][0]
                layers[27,i,j,k] = self.use_CA_for_indexes[counter][1]
                layers[28,i,j,k] = self.use_CA_for_indexes[counter][2]
                
                layers[29,i,j,k] = self.use_CB_for_indexes[counter][0]
                layers[30,i,j,k] = self.use_CB_for_indexes[counter][1]
                layers[31,i,j,k] = self.use_CB_for_indexes[counter][2]
                
            elif self.layers_use == 29:
                if self.layer_29_type == 0:
                    layers[26,i,j,k] = self.use_CA_for_indexes[counter][0]
                    layers[27,i,j,k] = self.use_CA_for_indexes[counter][1]
                    layers[28,i,j,k] = self.use_CA_for_indexes[counter][2]
                elif self.layer_29_type == 1:
                    layers[26,i,j,k] = self.use_CB_for_indexes[counter][0]
                    layers[27,i,j,k] = self.use_CB_for_indexes[counter][1]
                    layers[28,i,j,k] = self.use_CB_for_indexes[counter][2]
                else:
                    layers[26,i,j,k] = self.unit_v[counter][0]*self.use_CA_vetor
                    layers[27,i,j,k] = self.unit_v[counter][1]*self.use_CA_vetor
                    layers[28,i,j,k] = self.unit_v[counter][2]*self.use_CA_vetor
                    
                    


            ## just checking^^^^
            
            if self.use_res_index ==1:
                # amino acid number for connnectivity recovery starts from 1
                # do not use for training
                layers[-2,i,j,k] = static_data_for_res[7]
                layers[-1,i,j,k] = static_data_for_res[8]
            # interaction layer  
            #print(self.all_interacting_aa_from_all_atoms)
            # interaction_voxel = self.only_protein_residues[counter+1]   
            # print(mask.shape)
            mask[0,i,j,k] = static_data_for_res[-1]
            
            if self.return_res_numbers_from_1 == 1:
                res_mat_from_1[0,i,j,k] = static_data_for_res[0]
            
        self.voxel_mask = mask
        self.voxel_layers = layers
        if self.return_res_numbers_from_1 == 1:
            self.res_mat_from_1 = res_mat_from_1
    # def coordinates_for_ground_truth_and_predicted(self):
    #     xyz_cb_true = []
    
 
def is_model_params_correct():
    if not os.path.exists(config1.model_root_directory):
        print("Please set the correct model_root_directory in settings.py")
        return 0
    return 1
     


                  
if __name__ == "__main__":  
    from data_preparation.pdb_2_interaction_file_converter import pdb_to_interaction_file
    dir_ = "/home/sudhanshu/HDD3/Data/Sudhanshu/pdb_chains/dataset/pdbs/bad/pdb_npz/"
    for i in range(1):
       
        print(i)
        
        pdb_file = '/home/sudhanshu/HDD3/Data/Sudhanshu/pdb_chains/dataset/pdbs/bad/1M1J_3_000.pdb'
           
        #s,x=load_pdb_npz_to_static_data_and_coordinates(dir_ + '1M1J_3_000_data.pdb.npz')
        f = pdb_to_interaction_file(pdb_file, './',0, verbose=0)
        f.carb_aa_distance_calc_and_save = 1   
        f.save_data =0
        s,x = f.run_me() 
         
        
        for i in range(len(s[1])):
            s[1][i][-1]= 0
        
        
        xyz_vx = xyz_to_rotate_to_voxelize_to_translate()
        xyz_vx.max_pixel_translate_per_axis=0
        xyz_vx.max_rotation_plus_minus = 0
        xyz_vx.cube_start_points = 0
        xyz_vx.use_res_index =0
        f,u=xyz_vx.apply(x,s)
        
    
    # xyz_vx.pdb_f.read_pdb('/home/sudhanshu/HDD3/Data/Sudhanshu/pdb_chains/dataset/random_rotated/pdbs/test/CB_test.pdb')
    # xyz_vx.pdb_f.dump_pdb('./temp_files/m0.pdb')
    # xyz_vx.pdb_f.xyz_data =xyz_vx.xyz_beta_data
    # xyz_vx.pdb_f.refresh_from_xyz()
    # xyz_vx.pdb_f.dump_pdb('./temp_files/m1.pdb')
    # np.savez( './temp_files/xx.npz',
    #         layers = f)
    
    # np.savez('./temp_files/xx_mask.npz',
    #           interaction= u
    #           )
# print(np.max(f,0)-np.min(f,0))

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# plt.scatter(f[:,0],f[:,1],f[:,2])

