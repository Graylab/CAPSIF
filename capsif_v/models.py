#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 10:41:52 2021

@author: sudhanshu
Last update: 3/4/2023

Contains:
    UNet model, and model_train method.

"""

import torch
import torch.nn as nn
from tqdm import tqdm

def model_train(loader, model, optimizer, loss_fn, scaler, DEVICE):
    loop = tqdm(loader)
    train_loss = 0
    for batch_idx, (data, targets) in enumerate(loop):
        if data.shape[0] == 1:
            continue
        data = data.to(device=DEVICE,dtype=torch.float)
        targets = targets.float().to(device = DEVICE)

        #forward  
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(targets,predictions)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        train_loss += loss.item()*data.size(0)
        loop.set_postfix (loss = loss.item())  
    train_loss = train_loss/len(loader.sampler)
    return train_loss

        
class DoubleConv_3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv_3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 5,1,2,bias=False), 
            nn.BatchNorm3d(out_channels),  # trail change the sequence
            nn.ReLU(inplace=True),            
            nn.Conv3d(out_channels, out_channels, 5,1,2,bias=False),   
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        return self.conv(x)



# This form of model looks complex but give more flexibility for conv-size and channel choosing    
class UNET_3D(nn.Module):
    def __init__(self, in_channels = 4, out_channels = 1, ):
        super(UNET_3D, self).__init__()
        # self.conv_steps = [16,32,64,128,256] # for more steps
        self.conv_steps = [32,64,128,256]
        self.conv_steps = [in_channels] + self.conv_steps        
        self.in_channels = in_channels        
        self.pool_list = nn.ModuleList()
        self.double_conv = nn.ModuleList()
        self.double_conv_up = nn.ModuleList()
        self.upscale = nn.ModuleList()
        
        #pooling_array = [2,2,2,3,3]  # K-size for pooling for [72-36-18-9-3-1 scaling]
        pooling_array = [2,2,3,3]  # K-size for pooling for [36-18-9-3-1 scaling]
        pools = nn.ModuleDict() # more pooling methods can be added
        pools['2'] = nn.MaxPool3d (kernel_size = 2, stride = 2)
        pools['3'] = nn.MaxPool3d (kernel_size = 3, stride = 3)  
        #pools['4'] = nn.MaxPool3d (kernel_size = 4, stride = 4)  
        
        # Going Down
        for i in pooling_array:
            self.pool_list.append(pools[str(i)])         
            
        for down_step in range(len(self.conv_steps)-1):
            self.double_conv.append(DoubleConv_3D(self.conv_steps[down_step],self.conv_steps[down_step+1]))
        
        #BottleNeck
        self.double_conv.append(DoubleConv_3D(self.conv_steps[-1],self.conv_steps[-1]*2))
        
        # for upscaling
        pooling_array.reverse()  # for upscaling use
        self.conv_steps.reverse()
        
        for up_step in range(len(self.conv_steps[:-1])):
            self.upscale.append(
                nn.ConvTranspose3d(self.conv_steps[up_step]*2, 
                                       self.conv_steps[up_step], 
                                       kernel_size = pooling_array[up_step], 
                                       stride= pooling_array[up_step])
                )
            
            self.double_conv_up.append(
                DoubleConv_3D(self.conv_steps[up_step]*2, self.conv_steps[up_step])
            )
            
            
        self.conv_steps.reverse() # rereversed for use in forward
        #Final output convolution
        self.final_conv = nn.Conv3d(self.conv_steps[1], out_channels, kernel_size =1 )
        
        
    def forward(self,x):
        skips = []  # for jump connections
        
        # Down              
        for down_step in range(len(self.conv_steps)-1):
            x = self.double_conv[down_step](x)        
            skips.append(x)
            x = self.pool_list[down_step](x)        
        
        #bottle nexk entry layer
        x = self.double_conv[-1](x)        
        
        #Up Sacling
        self.conv_steps.reverse()
        skips.reverse()
        
        for up_step in range(len(self.conv_steps[:-1])):            
            x = self.upscale[up_step](x)   
            x = torch.cat((x, skips[up_step]), dim = 1)           
            x = self.double_conv_up[up_step](x)
                    
        x = self.final_conv(x)       
        x = torch.sigmoid(x)        
        return x
     
        
def test():
    x = torch.randn((2,1,72,72,72))
    #x = torch.randn((4,1,36,36,36))
    model = UNET_3D(in_channels=1, out_channels=1 )
    preds = model(x)
    print(x.shape, preds.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()