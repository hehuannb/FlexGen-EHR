import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import pandas as pd
from models.vae import VariationalAutoencoder, vae_loss_fn
seed = 804
torch.manual_seed(seed)


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
        return sample, stat, sample_y

def train_vae_stat(net, dataloader,  epochs=10):
    optim = torch.optim.Adam(net.parameters())

    for i in range(epochs):
        for _, batch,_ in dataloader:
            print(batch.shape)
            batch = batch.to(device)
            optim.zero_grad()
            x,mu,logvar = model(batch)
            loss = vae_loss_fn(batch, x, mu, logvar)
            loss.backward()
            optim.step()
            print(loss.item())
        # evaluate(validation_losses, net, test_dataloader, vae=True, title=f'VAE - Epoch {i}')
    torch.save(net, 'vae_stat.pt')

def train_vae_tmp(net, dataloader,  epochs=10):
    optim = torch.optim.Adam(net.parameters())

    for i in range(epochs):
        for batch, _,_ in dataloader:
            print(batch.shape)
            batch = batch.to(device)
            optim.zero_grad()
            x,mu,logvar = model(batch)
            loss = vae_loss_fn(batch, x, mu, logvar)
            loss.backward()
            optim.step()
            print(loss.item())
        # evaluate(validation_losses, net, test_dataloader, vae=True, title=f'VAE - Epoch {i}')
    torch.save(net, 'vae_tmp.pt')





if __name__ == "__main__":

    batch_size =  512
    device = torch.device("cuda")
    dataset_train_object = MIMICDATASET(x_path='m_train.csv',x_s_path='ms_train.csv',\
                                        y_path='my_train.csv', train=True, transform=False)
    samplerRandom = torch.utils.data.sampler.RandomSampler(data_source=dataset_train_object, replacement=True)
    train_loader = DataLoader(dataset_train_object, batch_size=batch_size,
                                shuffle=False, num_workers=2, drop_last=True, sampler=samplerRandom)

    ### Test data loader ####

    dataset_test_object = MIMICDATASET(x_path='m_test.csv',x_s_path='ms_test.csv',\
                                       y_path='my_test.csv',  train=False, transform=False)
    samplerRandom = torch.utils.data.sampler.RandomSampler(data_source=dataset_test_object, replacement=True)
    test_loader = DataLoader(dataset_test_object, batch_size=batch_size,
                                shuffle=False, num_workers=1, drop_last=True, sampler=samplerRandom)




    tmp_samples, sta_samples, y = next(iter(train_loader))
    feature_dim = sta_samples.shape[1]

    model = VariationalAutoencoder(feature_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_vae_stat(model, train_loader)
    # vae_sta = torch.load('vae_sta.pt')
    # vae_sta.eval()
    
    feature_dim = tmp_samples.shape[1]

    model = VariationalAutoencoder(feature_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_vae_tmp(model, train_loader)

    # vae_tmp = torch.load('vae_tmp.pt')
    # vae_tmp.eval()


