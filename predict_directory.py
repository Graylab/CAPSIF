#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 7 2023

@author: swcanner

 Current settings for B Factor visualization:
 `BFactor =  0.0` : Nonbinder
 `BFactor = 40.0` : CAPSIF:G Predicted Binder
 `BFactor = 59.9` : CAPSIF:V Predicted Binder
 `BFactor = 99.9` : CAPSIF:V and CAPSIF:G Predicted Binder


Usage: python predict_directory.py --input_dir [input_directory/default: 'sample_dir/']
    --v_model [CAPSIF:V Model/default:"./capsif_v/models_DL/my_checkpoint_best_36_2A_CACB_vector_coord_I_clean_data.pth.tar"]
    --g_model [CAPSIF:G Model/default:"./capsif_g/models_DL/cb_model.pth.tar"]
    --out [output_directory/defalt: 'sample_dir']
    --make_pdb [default: True]
Returns: "[output_directory]/capsif_predictions.txt" with a list of binding residues
    and "[output_directory]/*.pdb" with the pdbs with the BFactor identifying the binding Residues

"""

import sys
import os

def manage_flags(flags):
    n = len(flags)

    input_flags = ["--dir","--v_model",'--g_model','--out','--make_pdb','--help']

    input_dir = './sample_dir/'
    graph_model_dir = "./capsif_g/models_DL/cb_model.pth.tar"
    voxel_model_dir = "./capsif_v/models_DL/my_checkpoint_best_36_2A_CACB_vector_coord_I_clean_data.pth.tar"
    out_dir = "./sample_dir/output/"
    make_pdb = True;

    if n > 1:
        for kk in input_flags:
            if kk in flags:
                ind = flags.index(kk);

                if (kk == '--input_dir'):
                    input_dir = flags[ind+1]
                if (kk == '--v_model'):
                    voxel_model_dir = flags[ind+1]
                if (kk == '--g_model'):
                    graph_model_dir = flags[ind+1]
                if (kk == '--out'):
                    out_dir = flags[ind+1]
                if (kk == 'make_pdb'):
                    a = flags[ind+1]
                    a = a.upper()
                    if (a == "0" or a == "F" or a == "FALSE"):
                        make_pdb = False;
                if (kk == '--help'):
                    print("""
    Usage: python predict_directory.py --dir [input_directory/default: 'sample_dir/']
        --v_model [CAPSIF:V Model/default:"./capsif_v/models_DL/my_checkpoint_best_36_2A_CACB_vector_coord_I_clean_data.pth.tar"]
        --g_model [CAPSIF:G Model/default:"./capsif_g/models_DL/cb_model.pth.tar"]
        --out [output_directory/defalt: 'sample_dir/output/']
        --make_pdb [default: True]
    Returns: "[input_directory]/capsif_v.txt" and "[input_directory]/capsif_g.txt"
        and pdbs with the residue binding in the BFACTOR or TEMP column (if I got the col_name wrong - its the one that AF2 stores the pLDDT metric in...)
                    """)
                    exit()

    return input_dir, graph_model_dir, voxel_model_dir, out_dir, make_pdb

#barebones flag management
input_dir, graph_model_dir, voxel_model_dir, out_dir, make_pdb = manage_flags(sys.argv)

print("Using Directory: " + input_dir)
print("CAPSIF:V Model:  " + voxel_model_dir)
print("CAPSIF:G Model:  " + graph_model_dir)
print("Output Directory:" + out_dir)
print("Outputting PDBs: " + str(make_pdb))

input_dir = os.path.abspath(input_dir) + "/"
voxel_model_dir = os.path.abspath(voxel_model_dir)
graph_model_dir = os.path.abspath(graph_model_dir)
out_dir = os.path.abspath(out_dir) + "/"

#os.chdir("./capsif_v/")
#CAPSIF:V requirements
sys.path.append("./capsif_v/")
from capsif_v.utils import xyz_to_rotate_to_voxelize_to_translate
from capsif_v.prediction_utils import load_model as load_voxel_model, command_run
os.chdir('./capsif_v/')
from data_preparation.pdb_2_interaction_file_converter import pdb_to_interaction_file as capsif_v_pdb_preprocess
os.chdir('../')
sys.path.append('../')

os.chdir("./capsif_g/")
sys.path.append("./capsif_g/")
from capsif_g.dataset import load_predictor_model as load_graph_model
from notebook_library import download_pdb, predict_for_voxel as predict_voxel, visualize, preprocess_graph, predict_for_graph as predict_graph, output_structure_bfactor_biopython_BOTH as output_structure_bfactor
os.chdir("../")
sys.path.append("../")

import numpy as np
import time
import torch

#------

print("Intializing CAPSIF:V")
#Initialize Voxel

#Run only on pdb files
# Initialize pdb into npz file for reading
pdb_npz_file_reader = xyz_to_rotate_to_voxelize_to_translate()
pdb_npz_file_reader.max_rotation_plus_minus = 0
pdb_npz_file_reader.max_pixel_translate_per_axis = 0
pdb_npz_file_reader.use_res_index = 1
pdb_npz_file_reader.layers_use = 29
pdb_npz_file_reader.layer_29_type = 2
pdb_npz_file_reader.cube_start_points = 1
pdb_npz_file_reader.crop_extra_edge = 0
pdb_npz_file_reader.cube_start_points = 1

#load Voxel model
os.chdir('capsif_v')
start_time = time.time()
model = load_voxel_model(2,2,dir=voxel_model_dir)
model_time = time.time()
print("CAPSIF:V Model load time: ","%5.1f " % (model_time -start_time), "seconds.\n")
os.chdir('..')

#load Graph model
start_time = time.time()
graph_model = load_graph_model(graph_model_dir)
model_time = time.time()
print("CAPSIF:G Model load time: ","%5.3f " % (model_time -start_time), "seconds.")

#------

start_time = time.time();

ls = os.listdir(input_dir);

out_file = open(out_dir + 'capsif_predictions.txt','w+')
out_file.write("PDB_Name,CAPSIF:V_predictions,CAPSIF:G_predictions\n")

for pdb_file in ls:
    if '.pdb' in pdb_file:

        print('Currently treating: ' + pdb_file)
        os.chdir('capsif_v')
        f = capsif_v_pdb_preprocess( input_dir + pdb_file, input_dir,
                                            0, verbose=0,
                                            use_glycan=1)
        f.carb_aa_distance_calc_and_save = 1
        f.save_data =0
        s,x = f.run_me()

        if len(x) <= 1:
            if len(x) == 0:
                print("Can not read pdb file! Not a protein data.")

            if x[0] == -1:
                print("PyRosetta could not read glycan!")
                print("Check PDB file (ring glycan, clashes etc.)!")
                print("or use restart the code and run 'load_glycan_off' flag. Dice will be 0")

        #CAPSIF:V Predictions
        proteins,masks = pdb_npz_file_reader.apply(x,s)
        d,voxel_res,py_com,pred_vox=predict_voxel(torch.from_numpy(proteins), torch.from_numpy(masks), model,  2 ,save_npz=0)
        os.chdir("..")

        #CAPSIF:G Predictions
        os.chdir('capsif_g/')
        graph_dice, graph_results, graph_res = predict_graph(input_dir + pdb_file,model=graph_model)
        os.chdir('..')


        if make_pdb :
            output_structure_bfactor(in_file=input_dir + pdb_file,res_v=voxel_res,res_g=graph_res,out_file=out_dir + pdb_file[:pdb_file.index('.')] + "_predictions.pdb")

        out_file.write(pdb_file + " : " + voxel_res + '\n')

out_file.close()

end_time = time.time()

print("CAPSIF Predictions Finished!")
print("Took: ","%5.1f " % (end_time -start_time), "seconds.\n\n")
print("Outputted to: " + out_dir)

print("FIN")
