#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:43:19 2019

@author: huangjunkai
"""

import PIL.Image as Image
import numpy as np

img1 = Image.open("/home/cvlab04/Desktop/Code/Medical/u_net_liver/check/train/A001-2_instance-45_microbleed_2.png")
img1_np = np.array(img1) 
#print(img1_np)
img2 = Image.open("/home/cvlab04/Desktop/Code/Medical/u_net_liver/check/result/Threshold05_U_Net_trans_25_epoch_A001-2_instance-45.png")
img2_np = np.array(img2) 
#print(img2_np)

count1=np.count_nonzero(img1_np > 0)
count2=np.count_nonzero(img2_np > 0)
#print(count2)
ans=0
ans = np.count_nonzero(img1_np*img2_np>0)
'''
for i in range(len(img1_np)):
    for j in range(len(img1_np)):
        if img1_np[i][j]>0 and img2_np[i][j]>0:
            ans+=1
'''
dice_coe = (2*ans+0.0001)/(count1+count2 + 0.0001)
print("dice_coe:",dice_coe)
