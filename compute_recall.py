# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 18:03:45 2023

compute recall

@author: ASUS
"""

import numpy as np
from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import csv
import time
import math
from datetime import datetime
import json
import os





def compute_RecallRateAtN_forRange(all_retrievedindices_scores_allqueries, ground_truth_info):
    sampleNpoints = range(1,20) #Can be changed to range(1,0.1*len(all_retrievedindices_scores_allqueries[0])) for maximum N equal to 10% of the total reference images
    recallrate_values = np.zeros(len(sampleNpoints))
    itr=0
    for N in sampleNpoints:      
        recallrate_values[itr] = compute_RecallRateAtN(N, all_retrievedindices_scores_allqueries, ground_truth_info)
        itr=itr+1
    
    print(recallrate_values)
    return recallrate_values.tolist(), sampleNpoints

def compute_RecallRateAtN(N, all_retrievedindices_scores_allqueries, ground_truth_info):
    matches=[]
    total_queries=len(all_retrievedindices_scores_allqueries)
    match_found=0
    
    for query in range(total_queries):
        top_N_retrieved_ind=np.argpartition(all_retrievedindices_scores_allqueries[query], -1*N)[-1*N:]
        for retr in top_N_retrieved_ind:        
            if (retr in ground_truth_info[query][1]):
                match_found=1
                break

        if (match_found==1):
            matches.append(1)
            match_found=0
        else:
            matches.append(0)            
            match_found=0
     
    recallrate_at_N=float(np.sum(matches))/float(total_queries)
    
    return recallrate_at_N

def performance_comparison(scores_all): #dataset_directory has a numpy file named ground_truth_new.npy
    total_len = np.arange(len(scores_all)).reshape(-1,1)
    from sklearn.neighbors import NearestNeighbors   
    knn = NearestNeighbors(n_jobs=-1)
    knn.fit(total_len)
    dist, pos = knn.radius_neighbors(total_len,5)# ground truch range
    ground_truth_info = np.concatenate((total_len,pos.reshape(-1,1)),1)
    recallrate_at_K_range, sampleNpoints = compute_RecallRateAtN_forRange(scores_all, ground_truth_info) #K range is 1 to 20
    return recallrate_at_K_range