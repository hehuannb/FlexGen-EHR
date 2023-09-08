import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F


bce = torch.nn.L1Loss()

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_size):
        super(VariationalAutoencoder, self).__init__()
        self.fc1 = nn.Linear(input_size,512)
        self.fc21 = nn.Linear(512, 128)
        self.fc22 = nn.Linear(512, 128)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(128)

        self.bn3 = nn.BatchNorm1d(512)
        # self.bn4 = nn.BatchNorm1d(512)
        
        self.relu = nn.ReLU()
        
        self.fc3 = nn.Linear(128, 512)
        self.fc4 = nn.Linear(512, input_size)
    
    def encode(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        return self.bn2(self.fc21(x)), self.fc22(x)
        
    def decode(self, z):
        z = self.relu(self.bn3(self.fc3(z)))
        return self.fc4(z)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 *logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
        
    def forward(self,x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x = self.decode(z)
        return x, mu, logvar

def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1).float()  # Flatten
    m2 = target.view(num, -1).float()  # Flatten
    intersection = (m1 * m2).sum().float()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

# Reconstruction + KL divergence losses summed over all elements and batch
def vae_loss_fn(x, recon_x, mu, logvar, numeric=True):
    # if numeric:
    BCE = F.mse_loss(recon_x, x, reduction='sum')/9000
    # else:
        # recon_x[recon_x<0.5] = 0
        # recon_x[recon_x>=0.5] = 1
        # BCE = bce(recon_x, x)
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)\
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
