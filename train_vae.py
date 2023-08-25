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
        return sample, sample_y



def train_vae(net, dataloader,  epochs=10):
    optim = torch.optim.Adam(net.parameters())

    for i in range(epochs):
        for batch,_ in dataloader:
            batch = batch.to(device)
            optim.zero_grad()
            x,mu,logvar = model(batch)
            loss = vae_loss_fn(batch, x, mu, logvar)
            loss.backward()
            optim.step()
            print(loss.item())
        # evaluate(validation_losses, net, test_dataloader, vae=True, title=f'VAE - Epoch {i}')
    torch.save(net, 'vae_tmp.pt')


# def test():
#     model.eval()
#     test_loss = 0
#     with torch.no_grad():
#         for i, data in enumerate(test_loader):
#             data = data.to(device)
#             recon_batch, mu, logvar = model(data)
#             test_loss += loss_function(recon_batch, data, mu, logvar).item()

#     test_loss /= len(test_loader.dataset)
#     print('====> Test set loss: {:.4f}'.format(test_loss))

# def generate():
#     #####################################
#     #### Load model and optimizer #######
#     #####################################

#     # random seed
#     np.random.seed(1234)

#     # Loading the checkpoint
#     checkpoint = torch.load(os.path.join(args.MODELPATH, "model_epoch_1000.pth"))

#     # Setup model
#     model = VAE().to(device)

#     # in order to not to have a mismatch, if the training is done using multiple GPU,
#     # same should be done while testing.
#     # if cuda and opt.multiplegpu:
#     #     ngpu = 2
#     #     generatorModel = nn.DataParallel(model, list(range(ngpu)))

#     # Load models
#     model.load_state_dict(checkpoint['Generator_state_dict'])

#     # insert weights [required]
#     model.eval()

#     #######################################################
#     #### Load real data and generate synthetic data #######
#     #######################################################

#     # Load real data
#     real_samples = dataset_train_object.return_data()
#     num_fake_samples = 10000

#     # Generate a batch of samples
#     gen_samples = np.zeros_like(real_samples, dtype=type(real_samples))
#     n_batches = int(num_fake_samples / args.batch_size)
#     for i in range(n_batches):
#         random_input = torch.randn(args.batch_size, args.compress_dim).to(device)
#         sample = model.decode(random_input).cpu().data.numpy()
#         gen_samples[i * args.batch_size:(i + 1) * args.batch_size, :] = sample
#         # Check to see if there is any nan
#         assert (gen_samples[i, :] != gen_samples[i, :]).any() == False

#     gen_samples = np.delete(gen_samples, np.s_[(i + 1) * args.batch_size:], 0)
#     print(gen_samples.shape[0])
#     gen_samples[gen_samples >= 0.5] = 1.0
#     gen_samples[gen_samples < 0.5] = 0.0

#     # ave synthetic data
#     np.save(os.path.join(args.MODELPATH, "synthetic.npy"), gen_samples)

# def evaluate():

#     # Load synthetic data
#     gen_samples = np.load(os.path.join(args.MODELPATH, "synthetic.npy"), allow_pickle=True)

#     # Load real data
#     real_samples = dataset_train_object.return_data()[0:gen_samples.shape[0], :]

#     # Dimenstion wise probability
#     prob_real = np.mean(real_samples, axis=0)
#     prob_syn = np.mean(gen_samples, axis=0)

#     colors = (0, 0, 0)
#     plt.scatter(prob_real, prob_syn, c=colors, alpha=0.5)
#     x_max = max(np.max(prob_real), np.max(prob_syn))
#     x = np.linspace(0, x_max + 0.1, 1000)
#     plt.plot(x, x, linestyle='-', color='k')  # solid
#     plt.title('Scatter plot p')
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.show()

if __name__ == "__main__":

    batch_size =  512
    device = torch.device("cuda")
    dataset_train_object = MIMICDATASET(x_path='m_train.csv',y_path='my_train.csv', train=True, transform=False)
    samplerRandom = torch.utils.data.sampler.RandomSampler(data_source=dataset_train_object, replacement=True)
    train_loader = DataLoader(dataset_train_object, batch_size=batch_size,
                                shuffle=False, num_workers=2, drop_last=True, sampler=samplerRandom)

    ### Test data loader ####

    dataset_test_object = MIMICDATASET(x_path='m_test.csv',y_path='my_test.csv',  train=False, transform=False)
    samplerRandom = torch.utils.data.sampler.RandomSampler(data_source=dataset_test_object, replacement=True)
    test_loader = DataLoader(dataset_test_object, batch_size=batch_size,
                                shuffle=False, num_workers=1, drop_last=True, sampler=samplerRandom)

    random_samples, y = next(iter(test_loader))
    feature_dim = random_samples.shape[1]

    model = VariationalAutoencoder(feature_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)


    train_vae(model, train_loader)

    vae_tmp = torch.load('vae_tmp.pt')
    vae_tmp.eval()


