import sys
import os
sys.path.append(os.path.split(os.path.realpath(__file__))[0])
from utils import AttributeDict
import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.image as mpimg
from PIL import Image 
from copy import deepcopy
from sklearn.linear_model import LinearRegression
import heapq
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from MWSNet.model import siamso
import torch
from torchvision import transforms
from tqdm import tqdm


tf = transforms.Compose([  # str path -> img data
     transforms.ToTensor(),
      transforms.Resize((400,400)),
     transforms.Normalize((0.4962),(0.0423,)), # from imagenet,np: (0.),(4.5,), MIT :(0.4962),(0.0423,), self:(0.),(0.027,)
 ])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = siamso.SiamSO(1, 1).to(device)#Resnet.ResNet(4096, device)
model.load_state_dict(torch.load('./MWSNet/pretrain.mdl')['state_dict'])

class SeqLGPR():
    params = None
    
    def __init__(self, params):
        self.params = params

    def run(self):
        # begin with preprocessing of the images
        if (self.params.DO_PREPROCESSING) & (not self.params.differenceMatrix.load):
            results = self.doPreprocessing(self.params)
        
        # image difference matrix             
        if self.params.DO_DIFF_MATRIX:
            results = self.doDifferenceMatrix(results)
        
        # contrast enhancement    
        if self.params.DO_CONTRAST_ENHANCEMENT:
            results = self.doContrastEnhancement(results)        
        else:
            if self.params.DO_DIFF_MATRIX:
                results.DD = results.D
        
        # find the matches
        if self.params.DO_FIND_MATCHES:
            results = self.doFindMatches(results)
        return results
    
    def doPreprocessing(self,params):
        results = AttributeDict()
        results.dataset = []
        for i in range(len(self.params.dataset)):
            
            #### .npy file ####
            filename = '%s/preprocessing-%s%s.mat' % (self.params.dataset[i].savePath, self.params.dataset[i].saveFile, self.params.saveSuffix)     
            p = deepcopy(self.params)    
            p.dataset = self.params.dataset[i]
            d = AttributeDict()
            data = p.dataset.data[params.dataset[i].imageIndices]
            re_data = [] 
            for j in tqdm(range(len(data))):
                if self.params.DO_DEEPFEATURE:
                    re_data.append(SeqLGPR.feature_extractor(data[j].astype(np.float32)).detach().numpy().flatten('F'))
                else:
                    re_data.append(data[j].flatten('F'))
            d.preprocessing = np.array(re_data).T
            results.dataset.append(d)
            
            if self.params.dataset[i].preprocessing.save:
                results_preprocessing = results.dataset[i].preprocessing
                savemat(filename, {'results_preprocessing': results_preprocessing})

        return results
    
    @staticmethod
    def rgb2gray(rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    
   
    
    
    @staticmethod
    def patchNormalize(img, params):
        s = params.normalization.sideLength    
        
        n = np.arange(0, img.shape[0]+2, s)
        m = np.arange(0, img.shape[1]+2, s)
            
        for i in range(len(n)-1):
            for j in range(len(m)-1):
                p = img[n[i]:n[i+1], m[j]:m[j+1]]
                
                pp=np.copy(p.flatten())
                
                if params.normalization.mode != 0:
                    pp=pp.astype(float)
                    img[n[i]:n[i+1], m[j]:m[j+1]] = 127+np.reshape(np.round((pp-np.mean(pp))/np.std(pp, ddof=1)), (s, s))
                else:
                    f = 255.0/np.max((1, np.max(pp) - np.min(pp)))
                    img[n[i]:n[i+1], m[j]:m[j+1]] = np.round(f * (p-np.min(pp)))
                    
                #print str((n[i], n[i+1], m[j], m[j+1]))
        return img
    
    @staticmethod
    ## extract feature
    def feature_extractor(rawdata): #Takes in a list of reference images as outputs a list of feature descriptors corresponding to these images. 
        
        ref_desc = model.unet(tf(rawdata.astype("float32")).to(device).unsqueeze(0)) 
        ref_desc_cpu = ref_desc.cpu()
            
        return ref_desc_cpu
    
    def getDifferenceMatrix(self, data0preproc, data1preproc):
        # TODO parallelize 
        n = data0preproc.shape[1]#map
        m = data1preproc.shape[1]#query
        D = np.zeros((n, m))   
    
        for i in tqdm(range(n)):
            for j in range(m):
            # euclid distance
            # d = data1preproc.astype(np.int16) - np.tile(data0preproc[:,i],(m, 1)).T.astype(np.int16)
            # D[i,:] = np.sum(np.abs(d), 0)/n
            
            # cosine similarity
                # ref_desc = data0preproc[i].astype(np.float32)
                # query_desc = data1preproc[j].astype(np.float32)
                # d = np.dot(ref_desc.T,query_desc)/(0.0001+np.linalg.norm(query_desc)*np.linalg.norm(ref_desc))
                # D[i,j] = d
            
                
                ref_desc = data0preproc[:,i].astype(np.float32)
                query_desc = data1preproc[:,j].astype(np.float32)
            
                # NCC similarity
                d = self.NCC(ref_desc,query_desc)
                D[i,j] = d
    
        return D
    
    def NCC(self, a, b):
        '''similarity evaluate metric 1: NCC'''
        similarity = np.sum((a - np.mean(a))*(b- np.mean(b)))/(np.sqrt(np.sum((a - np.mean(a))**2)*np.sum((b - np.mean(b))**2))+0.00001)
        return similarity
    
    
    
    def doDifferenceMatrix(self, results):
        filename = '%s/difference-%s-%s.mat' % (self.params.savePath, self.params.dataset[0].imagePath[-8:], self.params.dataset[1].imagePath[-8:])  
    
        if self.params.differenceMatrix.load and os.path.isfile(filename):
            print('Loading image difference matrix from file %s ...' % filename)
    
            d = loadmat(filename)
            results.D = d['D']                                    
        else:
            if len(results.dataset)<2:
                print('Error: Cannot calculate difference matrix with less than 2 datasets.')
                return None
    
            print('Calculating image difference matrix ...')
    
            results.D = self.getDifferenceMatrix(results.dataset[0].preprocessing, results.dataset[1].preprocessing)
            
            # save it
            if self.params.differenceMatrix.save:                   
                savemat(filename, {'D':results.D})
            
        return results
    
    def enhanceContrast(self, D):
        # TODO parallelize
        DD = np.zeros(D.shape)
    
        for i in range(D.shape[0]):
            a=np.max((0, i-self.params.contrastEnhancement.R//2))
            b=np.min((D.shape[0], i+self.params.contrastEnhancement.R//2+1))
            v = D[a:b, :]
            DD[i,:] = (D[i,:] - np.mean(v, 0)) / np.std(v, 0, ddof=1)  
        
        return DD-np.min(np.min(DD))
    
    def doContrastEnhancement(self, results):
        
        filename = '%s/differenceEnhanced-%s-%s%s.mat' % (self.params.savePath, self.params.dataset[0]['imagePath'][-8:], self.params.dataset[1]['imagePath'][-8:])
        
        if self.params.contrastEnhanced.load and os.path.isfile(filename):    
            print('Loading contrast-enhanced image distance matrix from file %s ...' % filename)
            dd = loadmat(filename)
            results.DD = dd['DD']
        else:
            print('Performing local contrast enhancement on difference matrix ...')
               
            # let the minimum distance be 0
            results.DD = self.enhanceContrast(results.D)

            # save it?
            if self.params.contrastEnhanced.save:                        
                DD = results.DD
                savemat(filename, {'DD':DD})
                
        return results
    
    def doFindMatches(self, results):
     
        filename = '%s/matches-%s-%s%s.mat' % (self.params.savePath, self.params.dataset[0]['imagePath'][-8:], self.params.dataset[1]['imagePath'][-8:], self.params.saveSuffix)  

        
        print('Searching for matching images ...')
        
        # make sure ds is dividable by two
        self.params.matching.ds = self.params.matching.ds + np.mod(self.params.matching.ds,2)
    
        matches,seq_mat = self.getMatches(results.D)
               
        
        results.matches = matches
        results.seq_mat = seq_mat

        return results
    
    def getMatches(self, DD):
        # TODO parallelize
        matches = np.zeros((DD.shape[1],2))    
        seq_mat = np.zeros((DD.shape[0],DD.shape[1]))
        for N in range(self.params.matching.ds//2, DD.shape[1]-self.params.matching.ds//2):
            # find a single match
            
            # We shall search for matches using velocities between
            # params.matching.vmin and params.matching.vmax.
            # However, not every vskip may be neccessary to check. So we first find
            # out, which v leads to different trajectories:
                
            move_min = self.params.matching.vmin * self.params.matching.ds    
            move_max = self.params.matching.vmax * self.params.matching.ds    
            
            move = np.arange(int(move_min), int(move_max)+1)
            v = move.astype(float) / self.params.matching.ds #速度范围
            
            idx_add = np.tile(np.arange(0, self.params.matching.ds+1), (len(v),1))
            idx_add = np.floor(idx_add * np.tile(v, (idx_add.shape[1], 1)).T) # 定义对应y方向上不同速度的索引
            
            # this is where our trajectory starts
            n_start = N + 1 - self.params.matching.ds/2    
            x = np.tile(np.arange(n_start , n_start+self.params.matching.ds+1), (len(v), 1)) # query图像的idx  
            
            #TODO idx_add and x now equivalent to MATLAB, dh 1 indexing
            score = np.zeros(DD.shape[0])    
            
            # add a line of inf costs so that we penalize running out of data, 加inf作为跑出data外的惩罚
            DD = np.vstack((DD, -np.infty*np.ones((1,DD.shape[1]))))

            if self.params.DO_Rerank:
                past = matches[:N,0].reshape(-1,1)#max(0,N-15)
                index = np.arange(len(past)).reshape(-1,1)

                LR_model = LinearRegression().fit(index,past)
                delta = LR_model.coef_[0][0]

                fore_pos = past[-1] + delta
                top_index = np.argsort(DD[:,N])[-15:]

                weight = np.array([np.exp(-abs(fore_pos-x)/20)+1 for x in top_index])[:,0]
                DD[:,N][top_index] = weight*DD[:,N][top_index]
            
            ###
            y_max = DD.shape[0]        
            xx = (x-1) * y_max ##
            
            
            
            flatDD = DD.flatten('F')
            #遍历y轴，每个候选s都有v条轨迹,得到每条轨迹的分数和，对于query的每一帧遍历得到s*v个轨迹的分数
            for s in range(1, DD.shape[0]):   
                y = np.copy(idx_add+s)
                y[y>y_max] = y_max # 防止超出
                idx = (xx + y).astype(int) #得到展平后的索引
                ds = np.sum(flatDD[idx-1],1)
                score[s-1] = np.max(ds)
            
            ### get a sequence max score matrix
            seq_mat[:,N] = np.concatenate((np.zeros(self.params.matching.ds//2), score[:seq_mat.shape[0]-self.params.matching.ds//2]),axis = 0)#前面也补上
                        
            
            
            # find 3rd largest score 
            max_ind = np.where(score==heapq.nlargest(3,score)[0])
            sec_ind = np.where(score==heapq.nlargest(3,score)[1])
            
            
            # find min score and 2nd smallest score outside of a window
            # around the minimum 
            
            max_idx = np.argmax(score) #candidate
            max_value = score[max_idx]
            
            # false positive
            window = np.arange(np.max((0, max_idx-self.params.matching.Rwindow/2)), np.min((len(score), max_idx+self.params.matching.Rwindow/2)))
            not_window = list(set(range(len(score))).symmetric_difference(set(window))) #xor
            max_value_2nd = np.max(score[not_window])
            
            match = [max_idx + self.params.matching.ds/2, max_value]# min_idx + self.params.matching.ds/2 
            
            matches[N,:] = match
            
            
        empty_ind = np.where(seq_mat == 0)
        seq_mat[empty_ind] = DD[empty_ind]
        # for j in range(len(empty_ind[0])):
        #     y,x = empty_ind[0][j],empty_ind[1][j]
        #     seq_mat[y][x]=DD[y][x]
        ### Nan change to 0
        nan_ind = np.where(np.isnan(matches[:,0]))[0]
        matches[nan_ind] = [0,0]
        
        
        return matches,seq_mat
    
    