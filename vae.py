import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Encoder, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        
        # Calculate the size of the flattened features
        # For input size 128x128, after 4 conv layers with stride 2:
        # 128 -> 64 -> 32 -> 16 -> 8
        self.flattened_size = 256 * 8 * 8
        
        # Fully connected layers for mean and variance
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_var = nn.Linear(self.flattened_size, latent_dim)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Decoder, self).__init__()
        
        # Calculate the size of the flattened features
        self.flattened_size = 256 * 8 * 8
        
        # Fully connected layer
        self.fc = nn.Linear(latent_dim, self.flattened_size)
        
        # Transposed convolutional layers
        self.conv1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1)
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 256, 8, 8)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))
        
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var
    
    def loss_function(self, recon_x, x, mu, log_var):
        """
        Reconstruction + KL divergence losses summed over all elements and batch
        """
        # Ensure inputs are in range [0, 1]
        x = torch.clamp(x, 0, 1)
        recon_x = torch.clamp(recon_x, 0, 1)
        
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + KLD

def train_vae(model, train_loader, optimizer, device, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data)
        loss = model.loss_function(recon_batch, data, mu, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
    
    return train_loss / len(train_loader.dataset) 