from models import DiT_B_2
import torch
from torchvision import datasets, transforms
from tqdm import tqdm
from collections import OrderedDict
from copy import deepcopy
from Dataset import DiffusionDataset
import matplotlib.pyplot as plt
from torch.nn.parallel import DataParallel
from torch.serialization import add_safe_globals


add_safe_globals([DataParallel])

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

filepath='./LatentDiffusion/Dataset'
dataset_train = DiffusionDataset(filepath+"/Dataset_Train.h5")
train_loader = torch.utils.data.DataLoader(dataset=dataset_train,batch_size=64,shuffle=True)
filepath='./LatentDiffusion/Dataset'
dataset_val = DiffusionDataset(filepath+"/Dataset_Test.h5")
val_loader = torch.utils.data.DataLoader(dataset=dataset_val,batch_size=64,shuffle=True)

model_vae = torch.load(r'TopoFormer.pt', weights_only=False)
model_vae=model_vae.to(device)
model_vae.eval()

model=DiT_B_2(depth=12, hidden_size=768,in_channels=2,patch_size=2, num_heads=12,num_classes=130).to(device)
# checkpoint =torch.load("./checkpoint/pretrain.pt",weights_only=False).module.state_dict()
# model.load_state_dict(checkpoint)
model = torch.nn.DataParallel(model)

ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
requires_grad(ema, False)
# Prepare models for training:
update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
model.train()  # important! This enables embedding dropout for classifier-free guidance
ema.eval()  # EMA model should always be in eval mode

total_params = sum(p.numel() for p in model.parameters())

from diffusion import create_diffusion
diffusion=create_diffusion(timestep_respacing="",noise_schedule="squaredcos_cap_v2",diffusion_steps=1000)

n_epochs = 1000
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
losses = []
for epoch in range(n_epochs):
    epoch_loss = 0
    with tqdm(train_loader) as pbar:
        for i, (x, y) in enumerate(pbar):
            x = x[:,:,:128,:128].to(device)
            with torch.no_grad():
                _, _, latent_x = model_vae.module.encoder(x)
            y = y.to(device).to(torch.float).transpose(1,2)
            y=y[:,1:3,:].reshape(-1,1,130)/1000
            y=torch.abs(y)
            t = torch.randint(0, diffusion.num_timesteps, (latent_x.shape[0],), device=device)
            cond = dict(y=y)
            loss_dict = diffusion.training_losses(model, latent_x, t, cond)
            loss = loss_dict["loss"].mean()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            update_ema(ema, model)
            pbar.set_postfix(Epoch=epoch + 1, loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

        scheduler.step()

        average_loss = epoch_loss / len(train_loader)
        losses.append(average_loss)
        with open("losses.txt", "w") as f:
            for loss in losses:
                f.write(f"{loss}\n")

    if epoch % 10 == 0:
        torch.save(model, f"./checkpoint/model{epoch}_resume.pt")
        torch.save(ema, f"./checkpoint/model{epoch}_ema.pt")

torch.save(model, r"./checkpoint/diffusionmodel_resume.pt")
torch.save(ema, r"./checkpoint/diffusionmodel_ema.pt")

