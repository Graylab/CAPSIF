#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 15:35:24 2022
@author: sudhanshu
"""
import os
# from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import sys
# sys.path.insert(1, '/home/sudhanshu/HDD2/projects2/voxel_type_pc_interaction')




def load_npz_data_protein_(filename, mask_layer_present=False, layers=0):
    data=np.load(filename,allow_pickle=True)
    data = data['layers']
    req_shape = np.array(data.shape)    
    if mask_layer_present == True:       
        if layers == 0:
            req_shape[0] -= 1
        else:
            req_shape[0] = layers
            
    else:
        if layers != 0:
            req_shape[0] = layers
    
    out_data = np.zeros(req_shape)
    for i in range(req_shape[0]):
        out_data[i,...] = data[i,...]  
    return out_data


def load_npz_data_mask_(filename, field_name='interaction'):
   data=np.load(filename,allow_pickle=True)
   data = data[field_name]
   
   req_shape = np.array(data.shape)
   req_shape[0] =1
   
   out_data = np.zeros(req_shape)
   out_data[0,...] = data[-1,...]
   return out_data


def load_pdb_npz_to_static_data_and_coordinates(filename, filename2=None):
    # if two files are given. The coordinates from the first file
    #and chain id, interaction properties will be taken from 2nd file.
    #print(filename)
    data = np.load(filename,allow_pickle=True)
    CB_CA_xyz = data['all_res_CB_CA_xyz']
    res_parameters = data['all_res_fixed_data']
    
    if not filename2 == None:
        data2 = np.load(filename2,allow_pickle=True)
        res_parameters2 = data2['all_res_fixed_data']
        
        for i in range(len(res_parameters2[1])):
            res_parameters[1][i][-4:] = res_parameters2[1][i][-4:]
 
    
    return res_parameters, CB_CA_xyz



    # if transform is not None:
    #     transform.apply(CB_CA_xyz)
    
def load_npz_data_train_multilayer(filename, format_='torch'):
    data=np.load(filename,allow_pickle=True)
    data = data['layers']
    req_shape = np.array(data.shape)
    #req_shape[0] -= 1
    
    out_data = np.zeros(req_shape)
    # print(data.shape, out_data.shape)
    for i in range(req_shape[0]):
        out_data[i,...] = data[i,...]    
    
    if format_ == 'np':
        return out_data
    return torch.from_numpy(out_data)

def load_npz_data_mask_multilayer(filename, format_='torch',idx=-1):
    data=np.load(filename,allow_pickle=True)
    
    if data.files.count('interaction') == 1:        
        data = data['interaction']
        req_shape = np.array(data.shape)
        req_shape[0] =1
        
        out_data = np.zeros(req_shape)
        out_data[0,...] = data[-1,...]
        
    elif data.files.count('layers') == 1: 
        data = data['layers']
        req_shape = np.array(data.shape)
        req_shape[0] =1
        out_data = np.zeros(req_shape)
        
        out_data[0,...] = data[idx,...]
        
    if format_ == 'np':
        return out_data
    return torch.from_numpy(out_data)




class ThreeDDataset(Dataset):
    def __init__(self, protein_dir, mask_dir, transform=None, layers = 0):
        self.protein_dir = protein_dir
        self.mask_dir = mask_dir
        self.transform = transform  
        self.proteins = os.listdir(protein_dir)
        self.layers = layers
        #print("HIIIIIIIII",image_dir )

    def __len__(self): 
        return len(self.proteins)

    def __getitem__(self, index):        
        prt_path = os.path.join(self.protein_dir, self.proteins[index])
        mask_path = os.path.join(self.mask_dir, self.proteins[index].replace(".npz","_mask.npz"))        
        protein = load_npz_data_protein_(prt_path,mask_layer_present=False,layers=self.layers)
        mask = load_npz_data_mask_(mask_path)
        
        if self.transform is not None:
            augmenttations = self.transform.apply( protein, mask)
            protein = augmenttations[0]
            mask = augmenttations[1]

        return protein, mask
    
    
class PDB_NPZ_Dataset(Dataset):
    def __init__(self, protein_dir, transformer, layers = 0, train=0):
        self.protein_dir = protein_dir
        self.transformer = transformer
        self.proteins = os.listdir(protein_dir)
        self.layers = layers
        self.train = train
        #print("HIIIIIIIII",image_dir )

    def __len__(self): 
        return len(self.proteins)
    
    def __getitem__(self, index):  
        prt_path = os.path.join(self.protein_dir, self.proteins[index])
        static_data, xyz_data = load_pdb_npz_to_static_data_and_coordinates( prt_path )
        protein, mask = self.transformer.apply(xyz_data, static_data, self.train)
        return protein, mask
    
    

class ThreeDDataset2(Dataset):
    
    def __init__(self, protein_dir, mask_dir, transform=None, layers = 0):
        self.protein_dir = protein_dir
        self.mask_dir = mask_dir
        self.transform = transform  
        self.proteins = os.listdir(protein_dir)
        self.layers = layers
        #print("HIIIIIIIII",image_dir )

    def __len__(self): 
        return len(self.proteins)

    def __getitem__(self, index):        
        prt_path = os.path.join(self.protein_dir, self.proteins[index])
        mask_path = os.path.join(self.mask_dir, self.proteins[index].replace(".npz","_mask.npz"))        
        protein = load_npz_data_protein_(prt_path,mask_layer_present=False,layers=self.layers)
        mask = load_npz_data_mask_(mask_path)
        
        if self.transform is not None:
            augmenttations = self.transform.apply( protein, mask)
            protein = augmenttations[0]
            mask = augmenttations[1]

        return protein, mask

# def test():    
#     from settings import train_dir_protein, train_dir_mask
#     data = ThreeDDataset(train_dir_protein, train_dir_mask)
    
#     for i in data[0][0]:
#         print(np.mean(i))



# if __name__ == "__main__":
#     test()   