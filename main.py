from models.cvae import VariationalAutoencoder, vae_loss_fn
from models.ddpm import DDPM, ContextUnet
from models.meddiff import MedDiff
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import json
import pandas as pd
# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

class MIMICDATASET(Dataset):
    def __init__(self, x_t,x_s, ys, yt, train=None, transform=None):
        # Transform
        self.transform = transform
        self.train = train
        self.xt = x_t
        self.xs = x_s
        self.ys = ys
        self.yt = yt

    def return_data(self):
        return self.xt, self.xs, self.ys, self.yt

    def __len__(self):
        return len(self.xt)

    def __getitem__(self, idx):
        sample = self.xt[idx]
        stat = self.xs[idx]
        sample_ys = self.ys[idx]
        sample_yt = self.yt[idx]
        return sample, stat, sample_ys, sample_yt



if __name__ == '__main__':

    batch_size =  512
    device = torch.device("cuda")
    tasks = [
        'mortality_48h',
        'ARF_4h', 
        'ARF_12h',
        'Shock_4h',
        'Shock_12h',
    ]
    task = tasks[1]
    s = np.load('FIDDLE_eicu/features/{}/s.npz'.format(task))
    X = np.load('FIDDLE_eicu/features/{}/X.npz'.format(task))
    s_feature_names = json.load(open('FIDDLE_eicu/features/{}/s.feature_names.json'.format(task), 'r'))
    X_feature_names = json.load(open('FIDDLE_eicu/features/{}/X.feature_names.json'.format(task), 'r'))
    df_pop = pd.read_csv('FIDDLE_eicu/population/{}.csv'.format(task))
    x_s = torch.sparse_coo_tensor(torch.tensor(s['coords']), torch.tensor(s['data'])).to_dense().to(torch.float32)
    x_t = torch.sparse_coo_tensor(torch.tensor(X['coords']), torch.tensor(X['data'])).to_dense().to(torch.float32)
    x_t = x_t.sum(dim=1).to(torch.float32)

    idx_t = np.random.permutation(np.arange(x_t.shape[0]))[0: int(0.8 * x_t.shape[0])]
    idx_s = np.random.permutation(np.arange(x_s.shape[0]))[0: int(0.8 * x_s.shape[0])]
    x_s = x_s[idx_s]
    x_t = x_t[idx_t]
    y = torch.tensor(df_pop.ARF_LABEL.values).to(torch.float32)
    ys, yt = y[idx_s, idx_t]
    dataset_train_object = MIMICDATASET(x_t, x_s, ys, yt,\
                                         train=True, transform=False)
    train_loader = DataLoader(dataset_train_object, batch_size=batch_size, shuffle=True, \
                              num_workers=1, drop_last=False)

    
    tmp_samples, sta_samples, _ = next(iter(train_loader))
    feature_dim_s = sta_samples.shape[1]
    feature_dim_t = tmp_samples.shape[1]

    svae = VariationalAutoencoder(feature_dim_s).to(device)

    tvae = VariationalAutoencoder(feature_dim_t).to(device)
    tvae = torch.load('saved_models/vae_tmp.pt')
    svae = torch.load('saved_models/vae_stat.pt')
    svae.eval()
    tvae.eval()
    n_epoch = 1
    n_T = 50 
    device = "cuda"
    n_classes = 2
    n_feat = 256  
    lrate = 1e-4
    save_model = True
    w = 0.1
    ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=2), \
                betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.to(device)
    trainer = MedDiff(tvae, svae, ddpm,train_loader,epochs=n_epoch,\
                      model_path='Synthetic_MIMIC/diff.pt')
    trainer.generate(5000, 0)