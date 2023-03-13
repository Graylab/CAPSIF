#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 7 2023

@author: scanner1

Utility file for running both CAPSIF:V and CAPSIF:G in the notebooks and "predict_directory.py"
"""

#For prediction specific
import torch
import numpy as np
import os
import sys
import math
import py3Dmol
sys.path.append('caspif_v/')
from capsif_v.data_util import load_npz_data_mask_, load_npz_data_protein_
from capsif_v.prediction_utils import intersection , make_pymol_command
from capsif_v.utils import (
    load_checkpoint,
    dice,
)
from Bio.PDB import *
sys.path.append('../')

sys.path.append('capsif_g/')
from capsif_g.dataset import load_predictor_model, get_tpfp
from capsif_g.preprocess import pdb_to_interaction_file
sys.path.append("../")

#from settings import config1
from colorama import Fore, Style

def predict_for_voxel(protein_vox, real_mask_vox, model, model_type, save_npz=0, cutoff=0.5):

    chain_ids = ' ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    chimera_command =""
    x = protein_vox[:29,...].unsqueeze(0) #[:6,...]

    y = real_mask_vox
    pdb_aa_index = protein_vox[-2,...].unsqueeze(0).numpy()
    pdb_chain_index = protein_vox[-1,...].unsqueeze(0).numpy()
    #x = x.to(device,dtype=torch.float)

    #print(pdb_aa_index)
    #print(pdb_chain_index)

    preds1 = model(x.float())
    preds = (preds1 > cutoff)

    #print(preds1[preds])
    # print(preds[0])
    #x = x.to(device,dtype=torch.float)

    dice_score = dice(y,preds[0])

    #print(dice_score, preds[0].shape)
    predicted_index = torch.where(preds[0].squeeze(0) * pdb_aa_index> 0)
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


    ground_index = torch.where(y == 1)
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

    residues = Fore.GREEN + "Residues: "+Style.RESET_ALL+ pdb_sent[:-1] + " (PDB numbering)"
    pymol_command = make_pymol_command(ground_truth, pdb_sent, true_pos_str)

    #if nrg == 2:
    #    return dice_score, chimera_command

    return dice_score, pdb_sent[:-1], pymol_command, preds

#stolen from https://github.com/ProteinDesignLab/protein_seq_des/blob/master/seq_des/util/data.py
def download_pdb(pdb, data_dir, assembly=1):
    """Function to download pdb -- either biological assembly or if that
    is not available/specified -- download default pdb structure
    Uses biological assembly as default, otherwise gets default pdb.

    Args:
        pdb (str): pdb ID.
        data_dir (str): path to pdb directory
    Returns:
        f (str): path to downloaded pdb
    """
    if assembly:
        f = data_dir + "/" + pdb + ".pdb1"
        if not os.path.isfile(f):
            try:
                os.system("wget -O {}.gz https://files.rcsb.org/download/{}.pdb1.gz".format(f, pdb.upper()))
                os.system("gunzip {}.gz".format(f))
            except:
                f = data_dir + "/" + pdb + ".pdb"
                if not os.path.isfile(f):
                    os.system("wget -O {} https://files.rcsb.org/download/{}.pdb".format(f, pdb.upper()))
    else:
        f = data_dir + "/" + pdb + ".pdb"
    if not os.path.isfile(f):
        os.system("wget -O {} https://files.rcsb.org/download/{}.pdb".format(f, pdb.upper()))
    return f


def visualize(pdb_file,carb_res,r="a.b",width=600,height=500,colors=['lime','gray','purple']):

    with open(pdb_file) as ifile:
        system = "".join([x for x in ifile])

    view = py3Dmol.view(width=width, height=height)
    view.addModelsAsFrames(system)

    #print(r)
    if ("," in r):
        r = r.split(",")
    else:
        r = [r]

    i = 0
    for line in system.split("\n"):
        split = line.split()
        if len(split) == 0 or (split[0] != "ATOM" and split[0] != "HETATM"):
            continue
        if split[3] == "TIP3" or split[3] == "HOH":
            continue

        my_boi = split[5] + "." + split[4]
        idx = int(split[1])

        #show sidechains as sticks
        if (my_boi in r) and (split[2] != "N" and split[2] != "CA" and split[2] != "O" and split[2] != "C"):
            view.setStyle({'model': -1, 'serial': i+1}, {"stick": {'color': colors[0]}} )
        #color predicted backbone
        elif (my_boi in r):
            view.setStyle({'model': -1, 'serial': i+1}, {"cartoon": {'color': colors[0]}} )
        #color not-predicted backbone
        else:
            view.setStyle({'model': -1, 'serial': i+1}, {"cartoon": {'color': colors[1]}})

        #show the glycan in purple
        if my_boi in carb_res:
            view.setStyle({'model': -1, 'serial': i+1}, {"stick": {'color': colors[2]}} )


        i += 1
    view.zoomTo()
    view.show()




def preprocess_graph(file,train=0, randomize= 0):
    #
    f = pdb_to_interaction_file(file, None,0, verbose=0,saveData=0)
    features, xyz = f.run_me()

    if randomize == 1:
        xyz = xyz + (np.random.rand(xyz.shape[0], xyz.shape[1])*2 -1)*0.4 #randomizing by 0.2A

    edges1 = []
    edges2 = []
    N_xyz = xyz[0::4,:]
    CA_xyz = xyz[1::4,:]
    C_xyz = xyz[2::4,:]
    CB_xyz = xyz[3::4,:]
    xyz = CB_xyz
    nodes = torch.ones(len(xyz),1)
    edge_att =[]

    distance_cutoff = 12
    one_hot_aa = np.zeros(20)

    for i in range(len(xyz)):
        for j in range( len(xyz) ):
            ca_ca_dist = np.sqrt(sum((xyz[i] - xyz[j])**2))
            if  ca_ca_dist < distance_cutoff:
                edges1.append(i)
                edges2.append(j)
                edge_att.append([ca_ca_dist])

    edge_ = torch.stack([torch.LongTensor(edges1), torch.LongTensor(edges2)])

    out_features =[]
    ground_truth=[]

    if train == 0:
        pdb_chain = []
        pdb_res = []

    counter = 0
    for i in features[1]:
        #print(i)
        one_hot_aa = [0,]*20
        one_hot_aa[i[1]-1] = 1
        #print(i[3:8]," \t\t ", i[8],i[9])

        #out_features.append(i[2:-4] + one_hot_aa + list(XYZ[counter]))
        #Convert our raw angles to a continuous boi!!! - exclude omega
        sinner = np.sin( np.array([i[8],i[9]]) * np.pi / 180. )
        cousin = np.cos( np.array([i[8],i[9]]) * np.pi / 180. )

        #out_features.append(i[2:-4] + one_hot_aa + list(XYZ[counter]))
        out_features.append(sinner.tolist() + cousin.tolist() + i[3:8] + one_hot_aa )
        if train == 0:
            pdb_chain.append(i[11])
            pdb_res.append(i[10])

        ground_truth.append([i[-1]] )
        counter +=1

    out_features = torch.FloatTensor(out_features)
    ground_truth = torch.IntTensor( ground_truth )
    edge_attr = torch.FloatTensor(edge_att)


    if train == 1:
        return nodes, edge_, out_features, edge_attr, ground_truth
    else:
        return nodes, edge_, out_features, edge_attr, ground_truth, pdb_res, pdb_chain


def predict_for_graph(file_name, model=None, model_dir="./capsif_g/models_DL/cb_model.pth.tar", print_res=0,cutoff=0.5,DEVICE='cpu'):
    nodes, edge_, out_features, edge_attr, ground_truth, pdb_res, pdb_chain = preprocess_graph(file_name,0)

    if model == None:
        model = load_predictor_model(model_dir,DEVICE)


    nodes = nodes.to(device=DEVICE,dtype=torch.int)
    edge_ = edge_.to(device=DEVICE,dtype=torch.int)
    out_features = out_features.to(device=DEVICE,dtype=torch.float)
    edge_attr = edge_attr.to(device=DEVICE,dtype=torch.int)
    ground_truth = ground_truth.float().to(device = DEVICE)


    #print(out_features.shape,nodes.shape,edge_.shape,edge_attr.shape)
    predictions = model(out_features, nodes, edge_, edge_attr)
    #predictions = condense_(predictions[0],4,avg=True,max=False)
    predictions = predictions[0].detach().numpy()
    #ground_truth = torch.from_numpy(condense_(ground_truth,4,avg=False,max=True))
    #ground_truth = torch.from_numpy(ground_truth)
    preds = torch.from_numpy(predictions > cutoff).float()
    res=(np.where(preds.cpu()==1)[0])
    res_p=''
    groun_res=''

    y_true_f = ground_truth
    y_pred_f = preds

    converter = ' ABCDEFGHIJKLMNOPQ'

    for i in res:
        res_p = res_p +str(pdb_res[i])+ "." + converter[pdb_chain[i]]+","

    #print(preds.size(),ground_truth.size())

    #print(preds)
    #dice_v = dice(ground_truth,preds).item()
    smoothing_factor=0.01

    intersection = torch.sum(y_true_f * y_pred_f)
    dice_v = ((2. * intersection + smoothing_factor) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smoothing_factor))
    tpfp = get_tpfp(y_true_f,y_pred_f)

    out_str = 'DICE:' + "%4.4f, " % dice_v + 'TP:' + "%5d, " % tpfp[0] + 'FP:' + "%5d, " % tpfp[1] + 'FN:' + "%5d, " % tpfp[2] + 'TN:' + "%5d, " % tpfp[3]
    if print_res == 1:
        print(res_p[:-1])
    return dice_v, out_str, res_p[:-1]
    #return preds, ground_truth

def output_structure_bfactor_biopython(file,res,out_file):
    if (len(res) < 1):
        res = '-1.A'
    res = res.split(',')

    #Create a parser adn read the structures
    parser = PDBParser()
    data = parser.get_structure('CAPS',file)

    #go thru all chains and residues and atoms
    models = data.get_models()
    models = list(models)
    for m in range(len(models)):
        chains = list(models[m].get_chains())
        for c in range(len(chains)):
            residues = list(chains[c].get_residues())
            for r in range(len(residues)):
                #check if its a binding residue
                temp = 1.00
                #its a predicted residue -> BFactor = 99.99
                my_res = str(residues[r].id[1]).strip() + "." + str(chains[c].id).strip()
                if my_res in res:
                    temp = 99.99

                atoms = list(residues[r].get_atoms())
                for a in range(len(atoms)):
                    atoms[a].set_bfactor(temp)
                    #print(chains[c].id,residues[r].id[1],atoms[a].name)
    #output the file
    io = PDBIO()
    io.set_structure(data)
    io.save(out_file)

    return;


def output_structure_bfactor_biopython_BOTH(in_file,res_v,res_g,out_file,weights=[59.9,40.0]):
    if (len(res_v) < 1):
        res_v = '-1.A'
    if (len(res_g) < 1):
        res_g = '-1.A'
    res_g = res_g.split(',')
    res_v = res_v.split(',')

    #Create a parser adn read the structures
    parser = PDBParser()
    data = parser.get_structure('CAPS',in_file)

    #go thru all chains and residues and atoms
    models = data.get_models()
    models = list(models)
    for m in range(len(models)):
        chains = list(models[m].get_chains())
        for c in range(len(chains)):
            residues = list(chains[c].get_residues())
            for r in range(len(residues)):
                #check if its a binding residue
                temp = 0.00
                #its a predicted residue -> BFactor = 99.99
                my_res = str(residues[r].id[1]).strip() + "." + str(chains[c].id).strip()
                if my_res in res_v:
                    temp += weights[0]
                if my_res in res_g:
                    temp += weights[1]

                atoms = list(residues[r].get_atoms())
                for a in range(len(atoms)):
                    atoms[a].set_bfactor(temp)
                    #print(chains[c].id,residues[r].id[1],atoms[a].name)
    #output the file
    io = PDBIO()
    io.set_structure(data)
    io.save(out_file)

    return;

#Depricated - does not work :(
def output_structure_bfactor_rosetta(file,res,out_file):
    #print(res)
    if (len(res) < 1):
        res = '-1.A'
    res = res.split(',')

    pose = pose_from_pdb(file);

    s = pose.sequence()
    for r in range(1,len(s)+1):
        res_info = pose.pdb_info().pose2pdb(r)
        #print(res_info)
        curr_res = res_info.split(" ")
        #print(curr_res)
        rnum = str(int(curr_res[0]))
        chain = curr_res[1]
        my_res = rnum + "." + chain
        temp = 33
        if my_res in res:
            temp = 99.99

        for ii in range(1,pose.residue(r).natoms()+1):
            pose.pdb_info().bfactor(r,ii,temp)

        #print(my_res,pose.pdb_info().bfactor(r,1))


    new_pose = Pose()
    new_pose.assign(pose)
    #for r in range(1,len(s)+1):
    #    print(r,new_pose.pdb_info().bfactor(r,1))
    print(pose.pdb_info().obsolete())

    pose.dump_pdb(out_file)


    for r in range(1,len(s)+1):
        print(r,pose.pdb_info().bfactor(r,1))

    return;
