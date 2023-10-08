# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 16:35:54 2023


Train

@author: ASUS
"""
import datetime
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SubsetRandomSampler

from model import siamso,Resnet
from building_dataset import *

torch.manual_seed(2345)
np.random.seed(2345)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

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


def evaluate(net, loader,mode):
    # set loss function
    with torch.no_grad():
        net.eval()
        criterion = BCELogit_Loss    
        for iteration,(temp, search, label) in enumerate(loader):
            temp, search, label = temp.to(device),search.to(device),label.to(device)
            B, C, H, W = temp.shape
            # 输入网络
            match_map = net(temp, search)
            
            #计算loss
            loss = 0
            loss = criterion(match_map, label)
        
            del match_map,temp,search,label
        
    return loss


def train(net,
        device,
        epochs: int = 5,
        batch_size = 3,
        learning_rate: float = 1e-5,
        save_checkpoint: bool = True,
        ):
    # creat dataset
    train_db = LGPRdata('D:\Study\Code\LGPR\datasets\Train_data\self', 'train')
    val_db = LGPRdata('D:\Study\Code\LGPR\datasets\Train_data\self', 'val')
    test_db = LGPRdata('D:\Study\Code\LGPR\datasets\Train_data\self','test')

    train_loader = DataLoader(train_db, batch_size = batch_size, shuffle=True)
    val_loader = DataLoader(val_db, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(test_db, batch_size = batch_size, shuffle=True)

    # set loss function
    criterion = BCELogit_Loss
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1000, gamma = 0.5)

    
    # begin training
    nBatches = (len(train_db) + batch_size - 1) // batch_size
    best_loss = 1000; best_epoch = 0
    print('Start training....')
    total_time = time.time()
    for epoch in range(epochs):
        start = time.time()
        for iteration,(temp, search, label) in enumerate(train_loader):
            temp, search, label = temp.to(device),search.to(device),label.to(device)
            B, C, H, W = temp.shape
            # 输入网络
            match_map = model(temp, search)
            
            #计算loss
            optimizer.zero_grad()
            loss = 0
            loss = criterion(match_map, label)
            loss.backward()
            optimizer.step()
            # 更新学习率
            scheduler.step()

            if iteration % 5 == 0:
                print("==> Epoch[{}/{}]({}/{}): Loss: {:.4f}".format(epoch, epochs,iteration, 
                    nBatches, loss), flush=True)
        if epoch % 1 == 0:
            val_loss = evaluate(model, val_loader,'val')
            print("==> Epoch[{}/{}]  Validation Loss: {:.4f}".format(epoch, epochs, val_loss), flush=True)
            if val_loss < best_loss:
                print("Save model")

                best_epoch = epoch
                best_loss = val_loss
 
                # torch.save({'epoch': epoch,
                #         'state_dict': model.state_dict(),
                #         'optimizer' : optimizer.state_dict()}, 'epoch{}_loss{:.3f}.mdl'.format(epoch,best_loss))
                torch.save({'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict()}, 'bestUU.mdl')
            
            del loss, match_map  
            del temp, search, label

    print('best loss:', best_loss, 'best epoch:', best_epoch)
    
    model.load_state_dict(torch.load('best.mdl'))
    # torch.save(model, 'model.pkl')
    print('loaded from ckpt!')
    
    test_loss = evaluate(model, test_loader,'test')
    print('test loss:', test_loss)



    
    


    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = siamso.SiamSO(1,1).to(device)
    learning_rate = 1e-5
    num_epochs = 100
    batch_size = 4
    
    
    train(model, device, epochs=num_epochs, batch_size = batch_size, learning_rate=learning_rate)
            
            
        
        
        
        
        
        
