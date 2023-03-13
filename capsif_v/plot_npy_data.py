from data_util import load_pdb_npz_to_static_data_and_coordinates
import os
import numpy as np


types= ['train', 'val', 'test']

for t in types:
    files = os.listdir("../dataset/"+t+"/pdb_npz")


    count = [0,0]

    for i in files:
        if not i.endswith('npz'):
            continue
        count[0]+=1
        d=load_pdb_npz_to_static_data_and_coordinates("../dataset/"+t+"/pdb_npz/" + i)
        d = d[1][0::2,:]
        v=(max(np.max(d,0) - np.min(d,0)))
        if v > 72:
            print((np.max(d,0) - np.min(d,0)))
            count[1] += 1



    print(count, 100*count[1]/count[0])
