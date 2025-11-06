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
fig,axes=plt.subplots(1,10,figsize=(24,2))
for i in range(10):
    axes[i].imshow(dataset[selected_ids[i]-1][0][0])
plt.savefig('customer1.png', bbox_inches='tight',transparent=False)
plt.close(fig)

fig,axes=plt.subplots(1,10,figsize=(24,2))
for i in range(10,20):
    axes[i-10].imshow(dataset[selected_ids[i]-1][0][0])
plt.savefig('customer2.png', bbox_inches='tight',transparent=False)
plt.close(fig)

fig,axes=plt.subplots(1,10,figsize=(24,2))
for i in range(20,30):
    axes[i-20].imshow(dataset[selected_ids[i]-1][0][0])
plt.savefig('customer3.png', bbox_inches='tight',transparent=False)
plt.close(fig)

fig,axes=plt.subplots(1,10,figsize=(24,2))
for i in range(30,40):
    axes[i-30].imshow(dataset[selected_ids[i]-1][0][0])
plt.savefig('customer4.png', bbox_inches='tight',transparent=False)
plt.close(fig)

fig,axes=plt.subplots(1,10,figsize=(24,2))
for i in range(40,50):
    axes[i-40].imshow(dataset[selected_ids[i]-1][0][0])
plt.savefig('customer5.png', bbox_inches='tight',transparent=False)
plt.close(fig)