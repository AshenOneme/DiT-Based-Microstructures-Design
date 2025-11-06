import os
import numpy as np
import torch
import torch.nn as nn
from IPython.core.pylabtools import figsize
from select import select
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from collections import defaultdict
from Dataset import BinaryImageDataset
from tqdm import tqdm
import matplotlib.pyplot as plt


dataset = BinaryImageDataset('E:/LT/LatentDiffusion/PythonProject01/LatentDiffusion/Dataset_customer/Dataset_customer.h5')
selected_ids=np.loadtxt('selected_ids.txt',dtype=np.int32)
fig,axes=plt.subplots(20,10,figsize=(6,16))
k=0
for i in range(20):
    np.savetxt(f'./{i+1}/img.txt', dataset[selected_ids[k] - 1][0][0],fmt='%d')
    for j in range(10):
        axes[i,j].imshow(dataset[selected_ids[k]-1][0][0])
        axes[i,j].get_xaxis().set_visible(False)
        axes[i,j].get_yaxis().set_visible(False)
        k+=1



plt.show()
