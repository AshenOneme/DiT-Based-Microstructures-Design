import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from collections import defaultdict
from Dataset import BinaryImageDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from VAE import VAE_M
# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 配置参数
class Config:
    data_path = "Dataset_customer.h5"  # 包含images和ids的NPZ文件
    batch_size = 128  # 训练批量大小
    num_clusters = 1000  # 聚类数量
    pretrain_epochs = 1  # 预训练轮次
    save_interval = 1  # 模型保存间隔

model = VAE_M(device=device,z_channels=1).to(device)
checkpoint = torch.load("saved_models/ae_epoch50.pth")

model = torch.nn.DataParallel(model)
model.load_state_dict(checkpoint)

# model.encoder.load_state_dict(checkpoint["encoder"])
# model.decoder.load_state_dict(checkpoint["decoder"])

dataset = BinaryImageDataset('Dataset_customer.h5')

images=dataset[123][0].unsqueeze(0).to(device)

mu, sga, recon = model(images)
fig,axes=plt.subplots(1,2)
axes[0].imshow(images[0][0].detach().cpu().numpy())
axes[1].imshow(recon[0][0].detach().cpu().numpy())
plt.show()
