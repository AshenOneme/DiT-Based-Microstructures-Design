import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from collections import defaultdict
from Dataset import BinaryImageDataset
from tqdm import tqdm
from VAE import VAE_M
from sklearn.cluster import SpectralClustering
# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("saved_models", exist_ok=True)

# 配置参数
class Config:
    data_path = "E:\LT\LatentDiffusion\PythonProject01\LatentDiffusion\Dataset_customer\Dataset_customer.h5"  # 包含images和ids的NPZ文件
    batch_size = 128  # 训练批量大小
    num_clusters = 2500  # 聚类数量
    pretrain_epochs = 50  # 预训练轮次
    save_interval = 1  # 模型保存间隔

# 第二阶段：特征提取和K-means聚类
def apply_kmeans():
    # 加载预训练模型
    model = VAE_M(device=device,z_channels=1).to(device)
    checkpoint = torch.load("saved_models/ae_epoch50.pth")
    model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint)

    dataset = BinaryImageDataset('E:\LT\LatentDiffusion\PythonProject01\LatentDiffusion\Dataset_customer\Dataset_customer.h5')
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=False)

    # 提取特征和ID
    features = []
    all_ids = []
    model.eval()
    with torch.no_grad():
        for images, ids in tqdm(dataloader):
            images = images.to(device)
            _, _, z = model.module.encoder(images)
            z=z.view(-1,32*32)
            features.append(z.cpu().numpy())
            all_ids.extend(ids.numpy().tolist())

    features = np.concatenate(features, axis=0)
    all_ids = np.array(all_ids)

    np.save('features.npy',features)
    np.save('all_ids.npy', all_ids)

    features=np.load('features.npy')
    all_ids=np.load('all_ids.npy')

    # 执行K-means聚类
    kmeans =  MiniBatchKMeans(n_clusters=Config.num_clusters,batch_size=1280,verbose=1, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    # clusterer = SpectralClustering(n_clusters=Config.num_clusters,
    #                                affinity='nearest_neighbors',
    #                                n_neighbors=50,verbose=1)
    # cluster_labels = clusterer.fit_predict(features)

    # 保存聚类结果
    np.savez_compressed(
        "saved_models/kmeans_result.npz",
        cluster_labels=cluster_labels,
        # cluster_centers=kmeans.cluster_centers_,
        ids=all_ids
    )

# 主执行流程
if __name__ == "__main__":
    # 阶段2：执行K-means聚类
    print("\n===== Applying K-means Clustering =====")
    apply_kmeans()

