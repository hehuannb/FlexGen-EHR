import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch import nn, optim
from tqdm import tqdm
from models.vae import vae_loss_fn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda'
class MedDiff(nn.Module):
    def __init__(self, tvae, svae, ldm, trainloader,epochs=10):
        super(MedDiff, self).__init__()
        self.tvae = tvae
        self.svae = svae
        self.diff = ldm
        self.dopt = optim.Adam(self.diff.parameters(), lr=1e-4)
        self.train = trainloader
        self.n_epoch = epochs
        self.tvae.eval()
        self.svae.eval()
        self.xt = trainloader.dataset.xt
        self.xs = trainloader.dataset.xs

    def train_epoch(self):
        for ep in range(self.n_epoch):
            self.diff.train()
            print(f'epoch {ep}')
            # linear lrate decay
            self.dopt.param_groups[0]['lr'] = 1e-4*(1-ep/self.n_epoch)

            pbar = tqdm(self.train)
            for xt, xs, c in pbar:
                self.dopt.zero_grad()
                xt = xt.to(device)
                xs = xs.to(device)
                zt, _ =  self.tvae.encode(xt)
                zs, _ =  self.svae.encode(xs)
                z = torch.concat([zt, zs],dim=1)
                c = c.to(device)
                loss_d = self.diff(z, c)
                # z_gen, _ = self.diff.sample(xt.shape[0], (256,), device, label=c, guide_w=0.5)
                # z_t, z_s = torch.chunk(z_gen, 2, 1)
                # xt_gen = self.tvae.decode(z_t)
                # xs_gen = self.svae.decode(z_s)
                pbar.set_description(f"loss: {loss_d.item():.4f}")
                self.dopt.step()
            self.generate(20, 0)

    def generate(self, num_sample, label):
        self.diff.eval()
        with torch.no_grad():
            z_gen, _ = self.diff.sample(num_sample, (256,), device, label=[label], guide_w=0.5)
            z_t, z_s = torch.chunk(z_gen, 2, 1)
            xt_gen = self.tvae.decode(z_t)
            xs_gen = self.svae.decode(z_s)

        real_prob = np.mean(self.xt.cpu().detach().numpy(), axis=0)
        fake_prob = np.mean(xt_gen.cpu().detach().numpy(), axis=0)
        plt.scatter(real_prob, fake_prob)


        real_prob = np.mean(self.xs.cpu().detach().numpy(), axis=0)
        fake_prob = np.mean(xs_gen.cpu().detach().numpy(), axis=0)
        plt.scatter(real_prob, fake_prob)