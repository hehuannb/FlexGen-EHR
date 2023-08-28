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
from models.vae import VariationalAutoencoder
import matplotlib.pyplot as plt

class MIMICDATASET(Dataset):
    def __init__(self, x_path,x_s_path, y_path, train=None, transform=None):
        # Transform
        self.transform = transform
        self.train = train
        # load data here
        self.ehr = pd.read_csv(x_path,index_col=[0,1,2], header=[0,1,2])
        self.label =  pd.read_csv(y_path,index_col=[0,1,2])
        self.label['los_3'] = self.label['los_3'].astype('int8')
        self.label['los_7'] = self.label['los_7'].astype('int8')
        self.sta = pd.read_csv(x_s_path,index_col=[0,1,2], header=[0,1])
        self.sta.columns = self.sta.columns.droplevel()
        self.sta = pd.get_dummies(self.sta, columns = ['diagnosis'])
        self.sta = pd.get_dummies(self.sta, columns = ['ethnicity'])
        self.sta = pd.get_dummies(self.sta, columns = ['admission_type'])
        self.xt = torch.from_numpy(self.ehr.values).to(torch.float32)
        self.xs = torch.from_numpy(self.sta.values).to(torch.float32)
        self.y = torch.from_numpy(self.label.values)

        self.sampleSize = self.xs.shape[0]
        self.featureSize = self.xs.shape[1]

    def return_data(self):
        return self.ehr, self.label

    def __len__(self):
        return len(self.ehr)

    def __getitem__(self, idx):
        sample = self.xt[idx]
        stat = self.xs[idx]
        sample_y = self.y[idx]
        return sample, stat, sample_y[0]


# def generate(vae, diffusion, n_sample):
#     ddpm.load_state_dict(torch.load("ehrmodel_0.pth"))
#     ddpm.eval()
#     with torch.no_grad():
#         n_sample = 3000
#         x_gen, x_gen_store = ddpm.sample(n_sample, (10,), device, label=[0], guide_w=0.5)

#     df = pd.DataFrame(x_gen[:,:-1].cpu().numpy(), columns=x_train.columns[:-1])
#     fig, axes = plt.subplots(9, 1, figsize=(8, 25))
#     for i, c in enumerate(num_feat):
#         f = df[[c]].plot(kind='kde',ax=axes[i])
#         f = x_train[[c]].plot(kind='kde',color='red',ax=axes[i])

def generate(vae_t, vae_s, ddpm, column_t,column_s, n_sample, n_feat):
    ddpm.load_state_dict(torch.load("ehrmodel.pth"))
    ddpm.eval()
    #####################################
    #### Load model and optimizer #######
    #####################################

    with torch.no_grad():
        z_gen, _ = ddpm.sample(n_sample, (n_feat,), device, label=[0], guide_w=0.5)
        z_t, z_s = torch.chunk(z_gen, 2, 1)
        xt_gen = vae_t.decode(z_t)
        xs_gen = vae_s.decode(z_s)
    df_t = pd.DataFrame(xt_gen.cpu().numpy(), columns=column_t)
    df_s = pd.DataFrame(xs_gen.cpu().numpy(), columns=column_s)
    return df_t, df_s


if __name__ == "__main__":
    batch_size =  512
    device = torch.device("cuda")
    dataset_train_object = MIMICDATASET(x_path='m_train.csv',x_s_path='ms_train.csv',\
                                        y_path='my_train.csv', train=True, transform=False)
    samplerRandom = torch.utils.data.sampler.RandomSampler(data_source=dataset_train_object, replacement=True)
    train_loader = DataLoader(dataset_train_object, batch_size=batch_size,
                                shuffle=False, num_workers=2, drop_last=True, sampler=samplerRandom)

    n_epoch = 100
    batch_size = 64
    n_T = 50 
    device = "cuda"
    n_classes = 2
    n_feat = 256  
    lrate = 1e-4
    save_model = True
    # ws_test = [0.0, 0.5, 2.0] # strength of generative guidance
    w = 0.1

    ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=2), \
                betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.to(device)

    vae_tmp = torch.load('vae_tmp.pt').to(device)
    vae_tmp.eval()
    vae_sta = torch.load('vae_stat.pt').to(device)
    vae_sta.eval()


    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        pbar = tqdm(train_loader)
        loss_ema = None
        for xt, xs, c in pbar:
            optim.zero_grad()
                        
            xt = xt.to(device)
            xs = xs.to(device)
            zt, _ =  vae_tmp.encode(xt)
            zs, _ =  vae_sta.encode(xs)
            z = torch.concat([zt, zs],dim=1)
            c = c.to(device)
            loss = ddpm(z, c)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

    torch.save(ddpm.state_dict(), f"ehrmodel.pth")

    ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=2), \
                betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.to(device)
    column_name_tmp = dataset_train_object.ehr.columns
    column_name_sta = dataset_train_object.sta.columns
    df_t, df_s = generate(vae_tmp, vae_sta, ddpm, column_name_tmp, column_name_sta, \
                  n_sample=10, n_feat=n_feat)

    df_t.to_csv('synthetic_mimic/ldm_tmp.csv',  index=True)
    df_s.to_csv('synthetic_mimic/ldm_static.csv',  index=True)