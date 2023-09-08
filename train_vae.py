import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import os
import seaborn as sns
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

import pandas as pd
from models.vae import VariationalAutoencoder, vae_loss_fn
seed = 804
from sklearn.preprocessing import QuantileTransformer

class MIMICDATASET(Dataset):
    def __init__(self, x_t,x_s, y, train=None, transform=None):
        # Transform
        self.transform = transform
        self.train = train
        self.xt = x_t
        self.xs = x_s
        self.y = y
        self.sampleSize = self.xt.shape[0]

    def return_data(self):
        return self.xt, self.xs, self.label

    def __len__(self):
        return len(self.xt)

    def __getitem__(self, idx):
        sample = self.xt[idx]
        stat = self.xs[idx]
        sample_y = self.y[idx]
        return sample, stat, sample_y


def preprocessing(tmp_path, static_path, y_path):
    # read both temporal and static ehr featurs 
    tmp = pd.read_csv(tmp_path,index_col=[0,1,2], header=[0,1,2])
    xt_numeric = tmp.loc[:, tmp.columns.get_level_values(1)!='mask']

    # Create a GaussianQuantilesPreprocessing object
    qt = QuantileTransformer(n_quantiles=10, random_state=0, output_distribution='normal')

    # Fit the object to the data
    x_numeric = qt.fit_transform(xt_numeric.values)

    xt_numeric = torch.from_numpy(x_numeric).to(torch.float32)
    # xt_numeric = torch.nn.functional.normalize(xt_numeric, p=1, dim=0)
    # for i in range(x_t_numeric.shape[1]):
    #     x_t_numeric[:,i] = (x_t_numeric[:,i] -x_t_numeric[:,i].min())/(x_t_numeric[:,i].max() -x_t_numeric[:,i].min())
    xt_mask = tmp.loc[:, tmp.columns.get_level_values(1)=='mask']
    xt_mask = torch.from_numpy(xt_mask.values).to(torch.float32)
    xt = torch.concat([xt_mask, xt_numeric], 1)
    # shape of xt_mask, 2496, xt_numeric is 4992
    sta = pd.read_csv(static_path,index_col=[0,1,2], header=[0,1])
    # read target information which include mortality, length of stay
    label =  pd.read_csv(y_path,index_col=[0,1,2])
    label['los_3'] = label['los_3'].astype('int8')
    label['los_7'] = label['los_7'].astype('int8')
    # One hot diagnosis and others
    sta.columns = sta.columns.droplevel()
    sta = pd.get_dummies(sta, columns = ['diagnosis'])
    sta = pd.get_dummies(sta, columns = ['ethnicity'])
    sta = pd.get_dummies(sta, columns = ['admission_type'])

    xs = torch.from_numpy(sta.values).to(torch.float32)
    y = torch.from_numpy(label.values)
    return xt, xs, y

def train_vae_stat(net, dataloader,  epochs=30):
    optim = torch.optim.Adam(net.parameters())

    for i in range(epochs):
        running_loss = 0
        for _, batch,_ in dataloader:
            batch = batch.to(device)
            optim.zero_grad()
            x,mu,logvar = model(batch)
            loss = vae_loss_fn(batch, x, mu, logvar)
            loss.backward()
            optim.step()
            running_loss += loss.item()
        print(running_loss/512)
        # evaluate(validation_losses, net, test_dataloader, vae=True, title=f'VAE - Epoch {i}')
    torch.save(net, 'vae_stat.pt')

def train_vae_tmp(net, dataloader,  epochs=30):
    optim = torch.optim.Adam(net.parameters())
    for i in range(epochs):
        running_loss = 0
        for batch, _,_ in dataloader:
            batch = batch.to(device)
            optim.zero_grad()
            x,mu,logvar = model(batch)
            loss = vae_loss_fn(batch, x, mu, logvar)
            loss.backward()
            optim.step()
            running_loss += loss.item()
        print(running_loss/512)
        # evaluate(validation_losses, net, test_dataloader, vae=True, title=f'VAE - Epoch {i}')
    torch.save(net, 'vae_tmp.pt')





if __name__ == "__main__":

    batch_size =  512
    device = torch.device("cuda")
    x_t_path = 'm_train.csv'
    x_s_path = 'ms_train.csv'
    y_path = 'my_train.csv'
    x_t, x_s, y = preprocessing(x_t_path, x_s_path, y_path)

    dataset_train_object = MIMICDATASET(x_t, x_s, y, train=True, transform=False)
    train_loader = DataLoader(dataset_train_object, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)

    ### Test data loader ####
    ### DO NOT NEED this current stage
    # dataset_test_object = MIMICDATASET(x_path='m_test.csv',x_s_path='ms_test.csv',\
    #                                    y_path='my_test.csv',  train=False, transform=False)
    # samplerRandom = torch.utils.data.sampler.RandomSampler(data_source=dataset_test_object, replacement=True)
    # test_loader = DataLoader(dataset_test_object, batch_size=batch_size,
    #                             shuffle=False, num_workers=1, drop_last=False)

    tmp_samples, sta_samples, y = next(iter(train_loader))
    feature_dim_s = sta_samples.shape[1]
    feature_dim_t = tmp_samples.shape[1]

    model = VariationalAutoencoder(feature_dim_s).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_vae_stat(model, train_loader,epochs=60)
    vae_sta = torch.load('vae_stat.pt')
    vae_sta.eval()

    # model = VariationalAutoencoder(feature_dim_t).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # train_vae_tmp(model, train_loader,epochs=60)
    # vae_tmp = torch.load('vae_tmp.pt')
    # vae_tmp.eval()
    
    with torch.no_grad():
        # x_recon,mu,logvar = vae_tmp(x_t.cuda())
        x_recon_s,mu,logvar = vae_sta(x_s.cuda())
        x_recon_s[x_recon_s<0.5] = 0
        x_recon_s[x_recon_s>=0.5] = 1

        # real_prob = np.mean(x_t.cpu().detach().numpy(), axis=0)
        # fake_prob = np.mean(x_recon.cpu().detach().numpy(), axis=0)
        # plt.scatter(real_prob, fake_prob)


        real_prob = np.mean(x_s.cpu().detach().numpy(), axis=0)
        fake_prob = np.mean(x_recon_s.cpu().detach().numpy(), axis=0)
        plt.scatter(real_prob, fake_prob)
        # plt.xlim([0, 1])
        # plt.ylim([0, 1])

    # sns.set_style("whitegrid", {'grid.linestyle': ' '})
    # fig, ax = plt.subplots(figsize=(4.2, 3.8))
    # df = pd.DataFrame({'real': real_prob,  'fake': fake_prob, \
    #                    "feature": dataset_train_object.ehr.columns.get_level_values(2)})
    # sns.scatterplot(ax=ax, data=df, x='real', y='fake', hue="feature", s=10, \
    #                 alpha=0.8, edgecolor='none', legend=None, palette='Paired_r')
    # sns.lineplot(ax=ax, x=[0, 1], y=[0, 1], linewidth=0.5, color="darkgrey")
    # ax.set(xlabel="Bernoulli success probability of real data")
    # ax.set(ylabel="Bernoulli success probability of synthetic data")
    # ax.xaxis.label.set_size(8)
    # ax.yaxis.label.set_size(8)
    # ax.set_xlim([0, 1])
    # ax.set_ylim([0, 1])

