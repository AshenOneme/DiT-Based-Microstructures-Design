import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
import h5py
from UNet import ClassConditionedUnet_B
import torch.optim as optim
from Dataset import DiffusionDataset

filepath='./LatentDiffusion/Dataset'
dataset_train = DiffusionDataset(filepath+"/Dataset_Train.h5")
train_loader = torch.utils.data.DataLoader(dataset=dataset_train,batch_size=32,shuffle=True)
filepath='./LatentDiffusion/Dataset'
dataset_val = DiffusionDataset(filepath+"/Dataset_Test.h5")
val_loader = torch.utils.data.DataLoader(dataset=dataset_val,batch_size=32,shuffle=True)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_vae = torch.load(r'TopoFormer.pt', weights_only=False)
model_vae=model_vae.to(device)
model_vae.eval()

noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
# noise_scheduler = DDIMScheduler(num_train_timesteps=50)

n_epochs = 1000
loss_fn = nn.MSELoss()
net =  ClassConditionedUnet_B(in_channels=2,out_channels=2,cond_dim=65,cond_dim2=65).to(device)
net = torch.nn.DataParallel(net)
# Define optimizer
optimizer = optim.AdamW(net.parameters(), lr=1e-3)
# Define learning rate scheduler
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

losses = []

for epoch in range(n_epochs):
    epoch_loss = 0
    with tqdm(train_loader) as pbar:
        for i, (x, curves) in enumerate(pbar):
            x = x[:,:,:128,:128].to(device)
            y = curves[:,:,1].to(device).to(torch.float)/1000
            z = curves[:, :, 2].to(device).to(torch.float)/1000
            y,z = torch.abs(y),torch.abs(z)
            with torch.no_grad():
                _, _, latent_x = model_vae.module.encoder(x)
            noise = torch.randn_like(latent_x)
            timesteps = torch.randint(0, 999, (latent_x.shape[0],)).long().to(device)
            noisy_x = noise_scheduler.add_noise(latent_x, noise, timesteps)
            pred = net(noisy_x, timesteps,y,z)
            loss = loss_fn(pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(Epoch=epoch + 1, loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
    average_loss = epoch_loss / len(train_loader)
    losses.append(average_loss)    scheduler.step()

    with open("losses.txt", "w") as f:
        for loss in losses:
            f.write(f"{loss}\n")
    if epoch % 10==0:
        torch.save(net,f"./checkpoint/model{epoch}.pt")

torch.save(net, r"./checkpoint/diffusionmodel.pt")

