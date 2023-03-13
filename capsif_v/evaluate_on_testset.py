#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 12:08:20 2022
Last update: 3/11/2023

@author: sudhanshu

Usage: python evaluate_on_testset.py
Returns: Dice, DVO, and DCC for each file in ../dataset/test/pdb_npz/

"""
import torch
import os
import numpy as np
from prediction_utils import load_model, expression_compare2
import matplotlib.pyplot as plt
from utils import xyz_to_rotate_to_voxelize_to_translate, dice, is_model_params_correct

import sys
sys.path.append("..")
from data_util import load_pdb_npz_to_static_data_and_coordinates
from settings import config1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def predict_on_test_set( set_types=0, figure_name=None ):
    if not is_model_params_correct():
        return

    #set_types = 0 # test_set
    #          = 1 # alfafold test set
    #          = 2 # both

    models = [[2,0], #29 #CA
              [2,1], #29 #CB
              [2,2], #29 #vector CB CA << PROVIDED
              [3,-1],] #32 #coord CB CA

    layers_options = [6,26,29,32]  # ONLY PROVIDED 29

    out_result=[]

    pdb_npz_file_reader = xyz_to_rotate_to_voxelize_to_translate()
    pdb_npz_file_reader.max_rotation_plus_minus = 0
    pdb_npz_file_reader.max_pixel_translate_per_axis = 0
    pdb_npz_file_reader.cube_start_points=0 # otherwise random start for large proteins

    print(config1.dataset_dir)

    test_dir = config1.dataset_dir + "test/"  #test set
    test_dir_af = config1.dataset_dir + "test_af/" #alphafold set

    use_test_dir = test_dir
    if set_types ==1:
        use_test_dir = test_dir_af
        all_npz_test =[]
        for i in os.listdir(test_dir +"pdb_npz/"):
            if i.endswith("npz"):
                all_npz_test.append(i[:-8])
        all_npz_test.sort()

    all_npz_use =[]
    for i in os.listdir(use_test_dir +"pdb_npz/"):
        if i.endswith("npz"):
            all_npz_use.append(i[:-8])
    all_npz_use.sort()




    all_dice_array=[]
    for mt in [2,]:  #model_types

        model_type =mt
        model = load_model(models[model_type][0], models[model_type][1], DEVICE=DEVICE)

        pdb_npz_file_reader.layers_use = layers_options[models[model_type][0]]
        pdb_npz_file_reader.layer_29_type = models[model_type][1]
        pdb_npz_file_reader.return_res_numbers_from_1 =1

        bad_chains=[]
        good_chains = []
        all_dice = []
        all_DCC =[]
        all_DVO = []
        for name in all_npz_use:
            pdb_npz_file = use_test_dir +"pdb_npz/" +name + ".pdb.npz"


            static_data_zero, coordinates_zero = load_pdb_npz_to_static_data_and_coordinates(pdb_npz_file)
            static_data = static_data_zero
            coordinate_data = coordinates_zero

            if set_types ==1:  ## because modled pdbs do not have carbohydrate interactions
                test_pdb_npz = test_dir + "pdb_npz/" + name[:6]+'_data.pdb.npz'
                static_data, coordinate_data = load_pdb_npz_to_static_data_and_coordinates(
                    pdb_npz_file,test_pdb_npz)


            protein, real_mask = pdb_npz_file_reader.apply(coordinate_data,
                                                          static_data)

            preds1 = model(torch.from_numpy(protein).unsqueeze(0).float().to(DEVICE))
            preds = (preds1 > 0.5).cpu()

            #DVO and DCC____________________

            res_mat = pdb_npz_file_reader.res_mat_from_1
            ground_res_num = res_mat[0,...][np.where(real_mask[0,...]>0)].astype(int)-1
            pred_res_num = res_mat[0,...][np.where(preds[0][0,...]>0)].astype(int)-1

            #DCC
            coordinates_ground =  coordinate_data[ground_res_num,:]
            ground_COM = np.mean(coordinates_ground,0)

            coordinates_pred =  coordinate_data[pred_res_num,:]
            pred_COM = np.mean(coordinates_pred,0)

            DCC = np.sqrt(np.sum((ground_COM -pred_COM)**2))
            all_DCC.append( DCC)

            #DVO
            union_res = np.union1d(ground_res_num , pred_res_num)
            intersect_Res = np.intersect1d(ground_res_num , pred_res_num,assume_unique=1)
            DVO=0
            if len(union_res) >0:
                DVO=(len(intersect_Res)/len(union_res))
            all_DVO.append(DVO)



            #DVO and DCC____________________

            d = dice(torch.from_numpy(real_mask),preds[0])

            out_result.append([name,d.item(), DCC, DVO])
            # d=float(predict_for_protein_and_mask(protein, real_mask, model, model_type))
            all_dice.append(d)
            print(name, 'DICE:', "%4.2f, " % d.item(),'DCC:', "%4.2f, " % DCC,
                  'DVO:', "%4.2f, " % DVO)
            if d < 0.1:
                bad_chains.append(name)

            if d > 0.6:
                good_chains.append([name,d.item()])



        name = "Carb-UNet-3D"
        all_dice = np.array(all_dice)
        plt.hist(all_dice,20)
        plt.xlabel("Dice Score", fontsize=12)
        plt.ylabel("#Structures", fontsize=12)
        plt.xlim([0,1])
        plt.ylim([0,25])
        plt.title(name)
        h,xx=np.histogram(all_dice,20,(0,1))
        all_dice_array.append(h)
        if not figure_name == None:
            plt.savefig(figure_name, dpi=300 )
        # plt.savefig("Model_comapared_"+name+"_1.png", dpi=300 )


    plt.figure()
    model_names = [r'C$_\alpha$ Coords.',
                   r'C$_\beta$ Coords.',
                   r'C$_\alpha$-$C_\beta$ unit vector.',
                   r'C$_\alpha$ & $C_\beta$ Coords.']
    for i in all_dice_array:

        plt.plot((xx[0:-1]+xx[1:])/2, i,'-o')

    plt.legend(model_names)
    plt.xlabel("Dice Score",fontsize=12)
    plt.ylabel("#Structures",fontsize=12)
    plt.grid(1)

    plt.title(name)
    # plt.savefig("Model_comapared_"+name+"2.png", dpi=300 )
    return out_result
    

if __name__=="__main__":
    # os.chdir('..')
    fig_ext=''
    result_bound = predict_on_test_set(0, "./Figures/bound_histogram"+fig_ext+".png")
    plt.figure()
    result_unbound = predict_on_test_set(1,"./Figures/unbound_histogram"+fig_ext+".png")

    plt.figure()
    bound_dice = np.array([i[1] for i in result_bound])
    unbound_dice = np.array([i[1] for i in result_unbound])

    bound_DCC = np.array([i[2] for i in result_bound])
    unbound_DCC = np.array([i[2] for i in result_unbound])

    bound_DVO = np.array([i[3] for i in result_bound])
    unbound_DVO = np.array([i[3] for i in result_unbound])


    expression_compare2(bound_dice,unbound_dice,'Bound DICE', 'Unbound DICE',
                        "./Figures/bound_vs_ubound_test"+fig_ext+".png")


    print('Making plot for bound/unbound dice histogram in Figures directory')
    plt.figure()
    plt.hist(bound_dice,20)
    plt.hist(unbound_dice,20,(3.57015352e-04,1),alpha=0.5)
    plt.xlabel("Dice Score",fontsize=12)
    plt.ylabel("#Structures",fontsize=12)
    plt.legend(['bound','unbound'])
    plt.savefig("./Figures/bound_vs_ubound_test_hist"+fig_ext+".png", dpi=300)


    fraction=1

    DCC_cum_sum =[]
    DCC_cum_sum_unbound =[]
    for i in range(21*fraction):
        DCC_cum_sum.append(100*np.sum(bound_DCC<i*(1/fraction))/len(bound_DCC))
        DCC_cum_sum_unbound.append(100*np.sum(unbound_DCC<i*(1/fraction))/len(unbound_DCC))

    print('Making plot for bound/unbound DCC in Figures directory')
    plt.figure()
    plt.plot(np.arange(21*fraction)*(1/fraction),DCC_cum_sum)
    plt.plot(np.arange(21*fraction)*(1/fraction),DCC_cum_sum_unbound)
    plt.xlabel("DCC", fontsize=12)
    plt.ylabel("Success Rate (%)", fontsize=12)
    plt.xticks(range(0,21,2))
    plt.plot([4,4],[-10,110],'k--')
    plt.xlim([-1,21])
    plt.ylim([-5,99])
    plt.legend(['Bound','Unbound'],fontsize=14)
    plt.savefig("./Figures/bound_vs_ubound_DCC_"+fig_ext+".png", dpi=300)

    print('Making plot for bound/unbound DVO in Figures directory')
    plt.figure()
    plt.hist(bound_DVO,20,(0,1)),plt.hist(unbound_DVO,20,(0,1),alpha=0.5)
    plt.xlabel("DVO",fontsize=12)
    plt.ylabel("#Binding pockets",fontsize=12)
    plt.savefig("./Figures/bound_vs_ubound_DVO_"+fig_ext+".png", dpi=300)
