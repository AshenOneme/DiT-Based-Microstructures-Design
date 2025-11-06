import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from collections import defaultdict
from tqdm import tqdm

# 配置参数
class Config:
    data_path = "Dataset_customer.h5"  # 包含images和ids的NPZ文件
    batch_size = 128  # 训练批量大小
    num_clusters = 2500  # 聚类数量
    pretrain_epochs = 50  # 预训练轮次
    save_interval = 1  # 模型保存间隔


def sample_results():
    # 加载聚类结果
    data = np.load("saved_models/kmeans_result.npz")
    cluster_labels = data["cluster_labels"]
    all_ids = data["ids"].flatten()

    # 创建聚类到ID的映射
    cluster_dict = defaultdict(list)
    for cluster_id, img_id in zip(cluster_labels, all_ids):
        cluster_dict[cluster_id].append(img_id)

    # 从每个簇中采样（仅保留满足条件的簇）
    selected_ids = []
    valid_cluster_count = 0  # 统计有效聚类数量

    for cluster_id in range(Config.num_clusters):
        candidates = cluster_dict.get(cluster_id, [])

        # 严格筛选条件：仅保留样本数≥10的簇
        if len(candidates) >= 10:
            # 随机选择10个不重复样本
            selected = np.random.choice(candidates, 10, replace=False)
            selected_ids.extend(selected)
            valid_cluster_count += 1
            print(f"Cluster {cluster_id}: Selected 10 samples")
        else:
            # 直接跳过不符合条件的簇
            print(f"Cluster {cluster_id}: Discarded (only {len(candidates)} samples)")

    # 保存最终结果（保留所有符合条件的样本）
    selected_ids = np.array(selected_ids)
    np.savetxt("selected_ids.txt", selected_ids, fmt='%d')

    # 输出统计信息
    print(f"\n===== Final Result =====")
    print(f"Valid clusters: {valid_cluster_count}/{Config.num_clusters}")
    print(f"Total selected samples: {len(selected_ids)}")
    print(
        f"Average samples per cluster: {len(selected_ids) / valid_cluster_count:.1f}" if valid_cluster_count > 0 else "No valid clusters")


# 主执行流程
if __name__ == "__main__":
    print("\n===== Sampling Results =====")
    sample_results()
