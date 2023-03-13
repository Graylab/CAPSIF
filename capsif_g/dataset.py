#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 17:17:31 2022

@author: sudhanshu

CAPSIF:G dataset utility files

"""
import os
# from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import sys
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import egnn.egnn as eg
from preprocess import pdb_to_interaction_file

def load_pdb_npz_data_transform_and_cubify(filename):
    #print(filename)
    data = np.load(filename,allow_pickle=True)
    CB_CA_xyz = data['all_res_CB_CA_xyz']
    res_parameters = data['all_res_fixed_data']
    return res_parameters, CB_CA_xyz


def npz_to_graph_coverter2(file,train=1, randomize= 0):
    #
    features,xyz = load_pdb_npz_data_transform_and_cubify(file)
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

    #now we need to do a _repeat_ and _condense_ for our lovely atomTypes

    #print(out_features)
    #print(np.shape(out_features))
    #out_features = repeat_andAddOneHot(out_features,4)
    #print(np.shape(out_features))
    #print(out_features)
    out_features = torch.FloatTensor(out_features)

    #print(np.sum(np.array(ground_truth)))
    #ground_truth = repeat_(ground_truth,4);
    #print(np.sum(np.array(ground_truth)))
    #print(ground_truth)
    ground_truth = torch.IntTensor( ground_truth )
    edge_attr = torch.FloatTensor(edge_att)

    #print(nodes.size(),edge_.shape(),out_features.size(),ground_truth.size())

    if train == 1:
        return nodes, edge_, out_features, edge_attr, ground_truth
    else:
        return nodes, edge_, out_features, edge_attr, ground_truth, pdb_res, pdb_chain

class PDB_NPZ_Graph_Dataset(Dataset):
    def __init__(self, protein_dir, randomize = 0):
        self.protein_dir = protein_dir
        self.proteins = os.listdir(protein_dir)
        self.randomize = randomize
        #print("HIIIIIIIII",image_dir )

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, index):
        prt_path = os.path.join(self.protein_dir, self.proteins[index])
        nodes, edge_, out_features, edge_attr, ground_truth = npz_to_graph_coverter2(
            prt_path, train=1, randomize=self.randomize )
        #print(len(nodes), len(ground_truth))
        return nodes, edge_, out_features, edge_attr, ground_truth




def graph_data_loader( train_dir, val_dir, batch_size,
    num_workers=0, pin_memory=True, layers = 0 ):

    train_ds = PDB_NPZ_Graph_Dataset( protein_dir=train_dir, randomize=1  )
    train_loader = DataLoader( train_ds, batch_size=batch_size, num_workers=num_workers,
        pin_memory=pin_memory, shuffle=True )

    val_ds = PDB_NPZ_Graph_Dataset( protein_dir=val_dir)
    val_loader = DataLoader( val_ds, batch_size=1, num_workers=num_workers,
        pin_memory=pin_memory, shuffle=False )

    return train_loader, val_loader



from tqdm import tqdm

def model_train(loader, model, optimizer, loss_fn, scaler, DEVICE):
    loop = tqdm(loader)
    #print("HIIIIIIIII")
    all_loss = []
    for batch_idx, (nodes, edge, out_features, edge_attr, targets) in enumerate(loop):
        nodes = nodes.to(device=DEVICE,dtype=torch.int)
        edge = edge.to(device=DEVICE,dtype=torch.int)
        out_features = out_features.to(device=DEVICE,dtype=torch.float)
        edge_attr = edge_attr.to(device=DEVICE,dtype=torch.int)
        targets = targets.float().to(device = DEVICE)

        #forward


        optimizer.zero_grad()
        predictions = model(out_features[0], nodes[0], edge[0], edge_attr[0])

        #predictions = model(nodes[0], [ edge[0][0],edge[1][0]], edge_attr[0])
        # print(predictions, targets[0].shape)
        if loss_fn == dice_loss:
            loss = loss_fn(predictions[0],targets[0])
        else:
            loss = loss_fn(predictions[0],targets[0], None,None)
        if type(loss) == tuple:
            loss = loss[0]
        all_loss.append(loss)
        #print(len(targets))
        loss.backward()
        optimizer.step()




        # with torch.cuda.amp.autocast():
        #     predictions = model(out_features[0], nodes[0], [ edge[0][0],edge[1][0]], edge_attr[0])
        #     loss = loss_fn(targets,torch.sigmoid(predictions[0]))

        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        # optimizer.zero_grad()
        temp_loss = torch.FloatTensor(all_loss)
        temp_loss = torch.sum(temp_loss)/len(all_loss)
        loop.set_postfix (loss = temp_loss.item())
    return temp_loss.item()

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


def calc_loss(loader, model, DEVICE="cuda"):
    dice_score = 0
    model.eval()
    cm = np.zeros((2,2))
    with torch.no_grad():
        for count, (nodes, edge, out_features, edge_attr, targets) in enumerate(loader):

            nodes = nodes.to(device=DEVICE,dtype=torch.int)
            edge = edge.to(device=DEVICE,dtype=torch.int)
            out_features = out_features.to(device=DEVICE,dtype=torch.float)
            edge_attr = edge_attr.to(device=DEVICE,dtype=torch.int)
            targets = targets.float().to(device = DEVICE)


            # x = x.to(device,dtype=torch.float)
            # y = y.to(device, dtype=torch.float).unsqueeze(1)

            #preds = torch.sigmoid(model(x))
            predictions = model(out_features[0], nodes[0], edge[0], edge_attr[0])
            #predictions = model(nodes[0], [ edge[0][0],edge[1][0]], edge_attr[0])
            #print(preds)
            preds = (predictions[0] > 0.5).float()
            #preds = predictions[0];
            dice_score +=dice(targets[0],preds)
            #print(targets[0].shape, preds.shape)
            cm += confusion_matrix(targets[0],preds)
    print(cm)
    model.train()
    return dice_score/(count+1)

def load_predictor_model(model_dir="./models_DL/cb_model.pth.tar",DEVICE='cpu'):

    batch_size = 1
    n_nodes = 4
    n_feat = 29
    x_dim = 1
    n_layers = 8

    model = eg.EGNN2(in_node_nf=n_feat, hidden_nf=n_feat*2, out_node_nf=1,
                      in_edge_nf=1, n_layers=n_layers, attention=1,normalize=1).to(DEVICE)

    checkpoint=torch.load(model_dir)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model


def get_tpfp(y_true, y_pred, smoothing_factor=.01):
    y_true_f = y_true
    y_pred_f = y_pred

    # print(y_true_f, y_pred_f,"----")
    tp = torch.sum(y_true_f * y_pred_f)
    #1 or 0 -> get the difference between the two
    #FP => y_pred > y_true -> +1
    fp = torch.sum((y_pred_f.type(torch.int8)  - y_true_f.type(torch.int8)) == 1)
    fn = torch.sum((y_true_f.type(torch.int8) -  y_pred_f.type(torch.int8)) == 1)
    tn = len(y_true) - (tp + fp + fn);

    return [tp,fp,fn,tn]

def predict_for_npz_file(file_name, model=None, print_res=1,cutoff=0.5):
    DEVICE = 'cpu'
    nodes, edge_, out_features, edge_attr, ground_truth, pdb_res, pdb_chain = npz_to_graph_coverter2(file_name,0)

    if model == None:
        model = load_predictor_model()


    nodes = nodes.to(device=DEVICE,dtype=torch.int)
    edge_ = edge_.to(device=DEVICE,dtype=torch.int)
    out_features = out_features.to(device=DEVICE,dtype=torch.float)
    edge_attr = edge_attr.to(device=DEVICE,dtype=torch.int)
    ground_truth = ground_truth.float().to(device = DEVICE)


    #print(out_features.shape,nodes.shape,edge_.shape,edge_attr.shape)
    predictions = model(out_features, nodes, edge_, edge_attr)
    #predictions = condense_(predictions[0],4,avg=True,max=False)
    predictions = predictions[0].numpy()
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
    #print(dice_v)
    tpfp = get_tpfp(y_true_f,y_pred_f)

    print('DICE:', "%4.4f, " % dice_v,'TP:',"%5d, " % tpfp[0],
        'FP:',"%5d, " % tpfp[1],'FN:',"%5d, " % tpfp[2],
        'TN:',"%5d, " % tpfp[3])
    if print_res == 1:
        print(res_p[:-1])
    return dice_v
    #return preds, ground_truth


def predict_for_all_test(model_dir,test_dir,cutoff=0.5):
    import matplotlib.pyplot as plt
    dice_arr =[]
    model = load_predictor_model(model_dir)

    for i in os.listdir(test_dir):
        if not i.endswith("pdb.npz"):
            continue
        if "egnn" not in i:
            continue

        print(i, end = ' ')
        cur_dice = predict_for_npz_file(test_dir + i,model,0,cutoff=cutoff)
        dice_arr.append(cur_dice)
    print(np.mean(dice_arr))
    plt.hist(dice_arr)

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

def predict_directory(model_dir,test_dir,cutoff=0.5,DEVICE='cpu'):

    dice_arr =[]
    model = load_predictor_model(model_dir,DEVICE=DEVICE)
    DEVICE = torch.device(DEVICE)

    for file in os.listdir(test_dir):
        if not file.endswith('.pdb'):
            continue
        #preprocess the pdbs to npz's

        nodes, edge_, out_features, edge_attr, ground_truth, pdb_res, pdb_chain = preprocess_graph(test_dir + file)
        nodes = nodes.to(device=DEVICE,dtype=torch.int)
        edge_ = edge_.to(device=DEVICE,dtype=torch.int)
        out_features = out_features.to(device=DEVICE,dtype=torch.float)
        edge_attr = edge_attr.to(device=DEVICE,dtype=torch.int)
        ground_truth = ground_truth.float().to(device = DEVICE)
        #predict
        predictions = model(out_features, nodes, edge_, edge_attr)
        predictions = predictions[0].detach().numpy()
        preds = torch.from_numpy(predictions > cutoff).float()
        res=(np.where(preds.cpu()==1)[0])
        res_p=''
        groun_res=''

        y_true_f = ground_truth
        y_pred_f = preds

        converter = ' ABCDEFGHIJKLMNOPQ'

        for i in res:
            res_p = res_p +str(pdb_res[i])+ "." + converter[pdb_chain[i]]+","
        print(file,":",res_p)




def glycan_names():

    ms_type = { 0:'Glc',
                1:'Gal',
                2:'Man',
                3:'GlcNAc',
                4:'GalNac',
                5:'Fuc',
                6:'Xyl'}
