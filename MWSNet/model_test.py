# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 16:34:03 2023

model test

@author: ASUS
"""

from model import siamso,Resnet
import torch
import os, glob
import random, csv
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2
import torch.nn as nn
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from building_dataset import *

tf = transforms.Compose([  # str path -> img data
     transforms.ToTensor(),
     transforms.Resize((250,250)),
     transforms.Normalize((0.),(0.027,)), # from MIT (0.4962),(0.0423,), self
 ])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

db = LGPRdata('D:\Study\Code\LGPR\datasets\Train_data\self','train')

loader = DataLoader(db, batch_size = 1, shuffle=True)
net = siamso.SiamSO(1, 1).to(device)
net.load_state_dict(torch.load('D:/Study/Code/LGPR/LGPR-Bench-main/VPR_Techniques/deep_feature/bestUU.mdl')['state_dict']
                    ,strict = False)



raw = [];raw1 = []
net_feature = [];net1_feature = []
corr_map = []

# 加载数据
for i,(temp, search, label) in enumerate(loader):
    temp, search, label = temp.to(device),search.to(device),label.to(device)
    B, C, H, W = temp.shape
    
    # raw_data        
    match_map = net(temp,search)
    
    # feature
    raw.append(temp)
    raw1.append(search)

    net_feature.append(net.unet(temp))
    net1_feature.append(net.unet(search))

    corr_map.append(match_map)
    
    del match_map
    
    if i == 5:
        break




# In[2]
################## 特征图对比 ##################
group = 4

plt.figure();plt.imshow(raw[group][0][0].cpu().detach().numpy());plt.title('raw_data')
plt.figure();plt.imshow(raw1[group][0][0].cpu().detach().numpy());plt.title('raw_data1')

m_f = net_feature[group][0][0].cpu()
m_f_c = m_f.detach().numpy()
plt.figure();plt.imshow(m_f_c);plt.title('model_feature')
plt.figure();plt.imshow(net1_feature[group][0][0].cpu().detach().numpy());plt.title('model_feature1')


before_corr = F.conv2d(raw1[group][0][0].view(1, 1, raw1[group][0][0].shape[-2], raw1[group][0][0].shape[-1]),
                  weight = raw[group][0][0].view(-1, 1, raw[group][0][0].shape[-2], raw[group][0][0].shape[-1]),
                  groups = 1)


plt.figure();plt.imshow(before_corr[0][0].cpu().detach().numpy());plt.title('before corr map')
plt.figure();plt.imshow(corr_map[group][0][0].cpu().detach().numpy());plt.title('corr map')


# In[3]
############## 特征距离图前后对比 #################
# 原始数据
tsne = TSNE(n_components=3)
raw_ = raw[group]
raw_c = raw_.cpu().detach().numpy().squeeze(1).reshape(raw_.shape[0],-1)
raw_tsne = tsne.fit_transform(raw_c)


# 创建三维图形对象
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
point_size = 100
# 绘制三维散点图
ax.scatter(raw_tsne[:B,0], raw_tsne[:B,1], raw_tsne[:B,2]
           ,c = 'green', s = point_size, label = 'anchor')
ax.scatter(raw_tsne[B:int(B+nPos),0], raw_tsne[B:int(B+nPos),1],raw_tsne[B:int(B+nPos),2]
           ,c = 'red', s = point_size, label = 'positive')
ax.scatter(raw_tsne[int(B+nPos):int(B+nPos+nNeg),0], 
           raw_tsne[int(B+nPos):int(B+nPos+nNeg),1], raw_tsne[int(B+nPos):int(B+nPos+nNeg),2]
           ,c = 'blue', s= point_size, label = 'negative')

ax.set_xlabel('t-SNE Dimension 1')
ax.set_xlabel('t-SNE Dimension 2')
ax.set_xlabel('t-SNE Dimension 3')
ax.set_title('Feature Distance Visualization')

plt.show()


# In[3] 
# 提取特征后
# TSNE降维，将特征映射到3维空间
tsne = TSNE(n_components=3)
feature = net_feature[group]
feature_c = feature.cpu().detach().numpy().squeeze(1).reshape(feature.shape[0],-1)
feature_tsne = tsne.fit_transform(feature_c)


# 创建三维图形对象
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
point_size = 100
# 绘制三维散点图
ax.scatter(feature_tsne[:B,0], feature_tsne[:B,1], feature_tsne[:B,2]
           ,c = 'green', s = point_size, label = 'anchor')
ax.scatter(feature_tsne[B:int(B+nPos),0], feature_tsne[B:int(B+nPos),1], feature_tsne[B:int(B+nPos),2]
           ,c = 'red', s = point_size, label = 'positive')
ax.scatter(feature_tsne[int(B+nPos):int(B+nPos+nNeg),0], 
           feature_tsne[int(B+nPos):int(B+nPos+nNeg),1], feature_tsne[int(B+nPos):int(B+nPos+nNeg),2]
           ,c = 'blue', s= point_size, label = 'negative')

ax.set_xlabel('t-SNE Dimension 1')
ax.set_xlabel('t-SNE Dimension 2')
ax.set_xlabel('t-SNE Dimension 3')
ax.set_title('Feature Distance Visualization')

plt.show()