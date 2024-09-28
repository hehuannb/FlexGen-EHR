import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F


bce = torch.nn.BCEWithLogitsLoss()

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_size, hidden=256,out=64,numerical=True):
        super(VariationalAutoencoder, self).__init__()
        self.fc1 = nn.Linear(input_size+1,hidden)
        self.fc21 = nn.Linear(hidden, out)
        self.fc22 = nn.Linear(hidden, out)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(out)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(out+1, hidden)
        self.fc4 = nn.Linear(hidden, input_size)
        self.numeric=numerical
    
    def encode(self, x, cond):
        x = torch.hstack((x, cond.unsqueeze(1)))
        x = self.relu(self.bn1(self.fc1(x)))
        return self.bn2(self.fc21(x)), self.fc22(x)
        
    def decode(self, z, cond):
        cond_z = torch.hstack((z, cond.unsqueeze(1)))
        z = self.relu(self.bn3(self.fc3(cond_z)))
        return self.fc4(z)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 *logvar)
        eps = torch.randn_like(std)
        return eps.mul(std)+mu
        
    def forward(self,x, cond1):
        mu, logvar = self.encode(x, cond1)
        z = self.reparameterize(mu, logvar)
        x = self.decode(z, cond1)
        return x, mu, logvar, z

def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1).float()  # Flatten
    m2 = target.view(num, -1).float()  # Flatten
    intersection = (m1 * m2).sum().float()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

# Reconstruction + KL divergence losses summed over all elements and batch
def vae_loss_fn(x, recon_x, mu, logvar, numeric=True):
    if numeric:
        BCE = F.mse_loss(recon_x, x, reduction='sum')/9000
    else:
        # recon_x[recon_x<0.5] = 0
        # recon_x[recon_x>=0.5] = 1
        BCE = F.binary_cross_entropy_with_logits(recon_x, x)
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)\
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

