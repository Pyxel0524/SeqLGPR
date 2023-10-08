# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 11:24:00 2023

dataset

@author: ASUS
"""

import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import os, glob
import random, csv
import numpy as np
# import visdom
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from labels import *
from model import siamso
import torch.nn.functional as F

# def collate_fn(batch):
#     """Creates mini-batch tensors from the list of tuples (query, positive, negatives).
    
#     Args:
#         data: list of tuple (query, positive, negatives). 
#             - query: torch tensor of shape (3, h, w).
#             - positive: torch tensor of shape (3, h, w).
#             - negative: torch tensor of shape (n, 3, h, w).
#     Returns:
#         query: torch tensor of shape (batch_size, 3, h, w).
#         positive: torch tensor of shape (batch_size, 3, h, w).
#         negatives: torch tensor of shape (batch_size, n, 3, h, w).
#     """

#     batch = list(filter (lambda x:x is not None, batch))
#     if len(batch) == 0: return None, None, None, None, None

#     query, positives, negatives = zip(*batch)

#     query = data.dataloader.default_collate(query)
#     posCounts = data.dataloader.default_collate([x.shape[0] for x in positives])
#     negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives])
#     positives = torch.cat(positives, 0)
#     negatives = torch.cat(negatives, 0)
   
#     return query, positives, negatives, posCounts, negCounts

def BCELogit_Loss(score_map, labels):
    """ The Binary Cross-Correlation with Logits Loss.
    Args:
        score_map (torch.Tensor): The score map tensor of shape [B,1,H,W]
        labels (torch.Tensor): The label tensor of shape [B,H,W,2] where the
            fourth dimension is separated in two maps, the first indicates whether
            the pixel is negative (0) or positive (1) and the second one whether
            the pixel is positive/negative (1) or neutral (0) in which case it
            will simply be ignored.
    Return:
        loss (scalar torch.Tensor): The BCE Loss with Logits for the score map and labels.
    """
    labels = labels.unsqueeze(1)
    loss = F.binary_cross_entropy_with_logits(score_map, labels[:, :, :, :, 0],
                                              weight=labels[:, :, :, :, 1],
                                              reduction='mean')
    return loss


class LGPRdata(Dataset):

    def __init__(self, root, mode):
        super(LGPRdata, self).__init__()
        self.root = root
        # image, label
        self.temp_name = os.listdir(os.path.join(self.root, 'template'))
        self.search_name = os.listdir(os.path.join(self.root, 'search'))

        ## mode
        if mode == 'train':  # 80%
            self.temp_name = self.temp_name[:int(0.8 * len(self.temp_name))]
            self.search_name = self.search_name[:int(0.8 * len(self.search_name))]

        elif mode == 'val':  # 10% (80-90%)
            self.temp_name = self.temp_name[int(0.8 * len(self.temp_name)):int(0.9 * len(self.temp_name))]
            self.search_name = self.search_name[:int(0.8 * len(self.search_name))]

        elif mode == 'test':  # 10% (90-100%)
            self.temp_name = self.temp_name[int(0.9 * len(self.temp_name)):]
            self.search_name = self.search_name[:int(0.8 * len(self.search_name))]

    def __len__(self):
        return len(self.temp_name)

    def __getitem__(self, idx):
        
        tf = transforms.Compose([  # str path -> img data
            transforms.ToTensor(),
            transforms.Normalize((0.),(0.027,)), # 原来mean=0, std = 4.5，转成图片有变,self:(0.),(0.004,), MIT: (0.4962),(0.0423,)
        ])
        
        template = tf(Image.open(os.path.join(self.root, 'template',self.temp_name[idx])))
        search = tf(Image.open(os.path.join(self.root, 'search',self.search_name[idx])))
        label = torch.from_numpy(create_BCELogit_loss_label(label_size = 250, pos_thr = 20, neg_thr = 70))#mit: pos: 15, neg: 40
        # show positive/negative
        # plt.figure();plt.imshow(template)
        # plt.figure();plt.imshow(search)
        # plt.figure();plt.imshow(label[:,:,0])

        return template, search, label



def main():
    # import visdom
    import time
    criterion = BCELogit_Loss
    
    # using self implemented Dataset class
    db = LGPRdata('D:\Study\Code\LGPR\datasets\Train_data', 'train')
    # print(db.images[0], db.labels[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    model = siamso.SiamSO(1,1,device)
    # x, y = next(iter(db))
    # print(x.shape, y)

    # viz.image(db.denormalize(x), win='sample_x', opts=dict(title='sample_x'))

    loader = DataLoader(db, batch_size = 2, shuffle=True)
    
    for i,(temp, search, label) in enumerate(loader):
        B, C, H, W = temp.shape
        match_map = model(temp, search)
        criterion(match_map, label)
        
        plt.figure();plt.imshow(match_map[0,0].detach().numpy())
        plt.figure();plt.imshow(label[0,0].detach().numpy())


if __name__ == '__main__':
    main()