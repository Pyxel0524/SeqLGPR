# -*- coding: utf8 -*-

from parameters import defaultParameters
from utils import AttributeDict
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy
import time
import os

from seqLGPR import *
from compute_recall import *

if __name__ == "__main__":

    # set the parameters

    # start with default parameters
    params = defaultParameters()    
    
    # Nordland spring dataset
    ds = AttributeDict()
    ds.name = 'Route5_sun'
    
    path = 'D:/Study/Code/LGPR/datasets/Route_5/run_0056'
    
    ds.imagePath = path
    
    ds.prefix='run'#images-,run
    ds.extension='.npy'#.png,.npy
    ds.suffix=''
    ds.imageSkip = 20    # use every n-nth image, optical:100
    ds.imageIndices = range(1, 14000, ds.imageSkip) # 12500
    ds.savePath = 'results'
    ds.saveFile = '%s-%d-%d-%d' % (ds.name, ds.imageIndices[0], ds.imageSkip, ds.imageIndices[-1])
    ds.preprocessing = AttributeDict()
    ds.preprocessing.save = 1
    ds.preprocessing.load = 1 #1
    #ds.crop=[1 1 60 32]  # x0 y0 x1 y1  cropping will be done AFTER resizing!
    ds.crop=[]
    
    spring=ds

    ds2 = deepcopy(ds)
    # Nordland winter dataset
    ds2.name = 'Route5_Rainy'
    path = 'D:/Study/Code/LGPR/datasets/Route_5/run_0018'
    

    ds2.saveFile = '%s-%d-%d-%d' % (ds2.name, ds2.imageIndices[0], ds2.imageSkip, ds2.imageIndices[-1])
    ds2.imagePath = path

    # ds.crop=[5 1 64 32]
    ds2.crop=[]
    
    winter=ds2      

    params.dataset = [spring, winter]

    # using deep feature?
    params.DO_DEEPFEATURE = 1

    # load old results or re-calculate?
    params.differenceMatrix.load = 0
    params.contrastEnhanced.load = 0
    params.matching.load = 0

    # using rerank?
    params.DO_Rerank = 1

    # where to save / load the results
    params.savePath='results'
              
    ## now process the dataset
    t1=time.time()

    ds.data = np.load(os.path.join(ds.imagePath,'run.npy'))
    ds2.data = np.load(os.path.join(ds2.imagePath,'run.npy'))

    ss = SeqLGPR(params)  
    results = ss.run()
    t2=time.time()          
    print("time taken: " + str(t2-t1))
    
    ## compute recall 
    recallrate_at_K_range = performance_comparison(results.seq_mat)


