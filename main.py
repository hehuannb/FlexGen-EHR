from tqdm import tqdm
import torch
import numpy as np
import torchvision
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from models.ddpm import DDPM, ContextUnet
import matplotlib.pyplot as plt

class MIMICDATASET(Dataset):
    def __init__(self, x_path, y_path, train=None, transform=None):
        # Transform
        self.transform = transform
        self.train = train
        # load data here
        self.ehr = pd.read_csv(x_path,index_col=[0,1,2], header=[0,1,2])
        self.label =  pd.read_csv(y_path,index_col=[0,1,2])
        self.label['los_3'] = self.label['los_3'].astype('int8')
        self.label['los_7'] = self.label['los_7'].astype('int8')
        self.x = torch.from_numpy(self.ehr.values).to(torch.float32)
        self.y = torch.from_numpy(self.label.values)

        self.sampleSize = self.x.shape[0]
        self.featureSize = self.x.shape[1]

    def return_data(self):
        return self.ehr, self.label

    def __len__(self):
        return len(self.ehr)

    def __getitem__(self, idx):
        sample = self.x[idx]
        sample_y = self.y[idx]
        return sample, sample_y[0]

# hardcoding these here
if __name__ == "__main__":



    batch_size =  512
    device = torch.device("cuda")
    dataset_train_object = MIMICDATASET(x_path='m_train.csv',y_path='my_train.csv', train=True, transform=False)
    samplerRandom = torch.utils.data.sampler.RandomSampler(data_source=dataset_train_object, replacement=True)
    train_loader = DataLoader(dataset_train_object, batch_size=batch_size,
                                shuffle=False, num_workers=2, drop_last=True, sampler=samplerRandom)

    n_epoch = 50
    batch_size = 64
    n_T = 20 
    device = "cuda"
    n_classes = 2
    n_feat = 128  
    lrate = 1e-4
    save_model = True
    # ws_test = [0.0, 0.5, 2.0] # strength of generative guidance
    w = 0.1

    ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=2), \
                betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.to(device)

    # # optionally load a model
    # # ddpm.load_state_dict(torch.load("./data/diffusion_outputs/ddpm_unet01_mnist_9.pth"))
    vae_tmp = torch.load('vae_tmp.pt')
    vae_tmp.eval()


    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        pbar = tqdm(train_loader)
        loss_ema = None
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            loss = ddpm(x, c)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

    # torch.save(ddpm.state_dict(), f"ehrmodel_{ep}.pth")
    # ddpm.load_state_dict(torch.load("ehrmodel_0.pth"))
    # ddpm.eval()
    # with torch.no_grad():
    #     n_sample = 3000
    #     x_gen, x_gen_store = ddpm.sample(n_sample, (10,), device, label=[0], guide_w=0.5)

    # df = pd.DataFrame(x_gen[:,:-1].cpu().numpy(), columns=x_train.columns[:-1])
    # fig, axes = plt.subplots(9, 1, figsize=(8, 25))
    # for i, c in enumerate(num_feat):
    #     f = df[[c]].plot(kind='kde',ax=axes[i])
    #     f = x_train[[c]].plot(kind='kde',color='red',ax=axes[i])