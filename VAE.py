import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bs = 100

train_dataset = datasets.MNIST(root='./mnist_data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./mnist_data', train=False, transform=transforms.ToTensor(), download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size =bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=True)

class VAE(nn.Module):
    def __init__(self, x_dim, h1_dim, h2_dim, z_dim):
        super(VAE, self).__init__()

        #encoder
        self.fc1 = nn.Linear(x_dim, h1_dim)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        self.fc31 = nn.Linear(h2_dim, z_dim)
        self.fc32 = nn.Linear(h2_dim, z_dim)

        #decoder
        self.fc4 = nn.Linear(z_dim, h2_dim)
        self.fc5 = nn.Linear(h2_dim, h1_dim)
        self.fc6 = nn.Linear(h1_dim, x_dim)
    
    def encoder(self, x):
        h = F.relu(self.fc1(x.view(-1, 784)))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)
    
    def sampling_z(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu
    
    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h))
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampling_z(mu, log_var)
        return self.decoder(z), mu, log_var


vae = VAE(x_dim=784, h1_dim=512, h2_dim=512, z_dim=2)
vae.to(device)

optimizer = optim.Adam(vae.parameters())

def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction= 'sum')
    KLD = -0.5*torch.sum(1 + log_var - mu.pow(2) - torch.exp(log_var))
    return BCE + KLD


def train(epoch):
    vae.train()
    train_loss = 0
    for batch_id, (data, label) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        recon_batch, mu, log_var = vae(data)
        loss = loss_function(recon_batch, data, mu, log_var)
        loss.backward()

        train_loss += loss.item()
        optimizer.step()

        if batch_id % 100 == 0:
            print('Train epoch ',batch_id*len(data))
            print( 'Loss', loss.item()/len(data))
    
    print('Epoch: ', epoch)
    print('Average Loss: ', train_loss/len(train_loader.dataset))

def test():
    vae.eval()
    test_loss = 0

    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            recon_batch, mu, log_var = vae(data)

            test_loss += loss_function(recon_batch, data, mu, log_var)

    print('Test Loss: ', test_loss/len(test_loader))


def test_plot():
    vae.eval()
    test_loss = 0

    latent_global = {}
    

    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, log_var = vae(data)
            latent_vector = vae.sampling_z(mu, log_var)
            latent_vector = latent_vector.numpy()
            label = label.numpy()
            
            if i == 0:
                z_global = latent_vector
                l_global = label
            else:
                z_global = np.append(z_global, latent_vector, axis=0)
                l_global = np.append(l_global, label)
            test_loss += loss_function(recon_batch, data, mu, log_var)

    print('Test Loss: ', test_loss/len(test_loader))

    colors = np.argmax(label)

    plt.figure()
    plt.scatter(z_global[:,0], z_global[:,1], c=l_global)
    plt.title('Latent vector coordinates, both loss')
    plt.savefig('graph_both_loss.png')
    plt.show()


#For training and testing VAE use below commented code

# for epoch in range(1,51):
#     train(epoch)
#     test()



#Load existing model to test hypothesis. 
checkpoint = torch.load('./ckpt/vae.pth', map_location='cpu')
vae.load_state_dict(checkpoint)
test_plot()

#Save generated images
with torch.no_grad():
    z = torch.randn(64,2).to(device)
    sample = vae.decoder(z).to(device)
    save_image(sample.view(64,1,28,28), './sample_both_loss.png')