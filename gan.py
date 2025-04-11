import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (3) x 128 x 128
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 4 x 4
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        # Get batch size
        batch_size = input.size(0)
        # Forward pass through the network
        output = self.main(input)
        # Reshape to match batch size
        return output.view(batch_size, -1).mean(dim=1)

def train_gan(generator, discriminator, train_loader, g_optimizer, d_optimizer, device, num_epochs=1):
    generator.train()
    discriminator.train()
    
    for epoch in range(num_epochs):
        g_losses = []
        d_losses = []
        
        for batch_idx, (real_imgs, _) in enumerate(train_loader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            
            # Train Discriminator
            d_optimizer.zero_grad()
            
            # Real images
            output_real = discriminator(real_imgs)
            label_real = torch.ones(batch_size).to(device)
            d_loss_real = F.binary_cross_entropy(output_real, label_real)
            
            # Fake images
            noise = torch.randn(batch_size, 100, 1, 1).to(device)
            fake_imgs = generator(noise)
            output_fake = discriminator(fake_imgs.detach())
            label_fake = torch.zeros(batch_size).to(device)
            d_loss_fake = F.binary_cross_entropy(output_fake, label_fake)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # Train Generator
            g_optimizer.zero_grad()
            output_fake = discriminator(fake_imgs)
            g_loss = F.binary_cross_entropy(output_fake, label_real)
            g_loss.backward()
            g_optimizer.step()
            
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            
            if batch_idx % 10 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(real_imgs)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\\t'
                      f'D_loss: {d_loss.item():.6f}\\tG_loss: {g_loss.item():.6f}')
        
        print(f'====> Epoch: {epoch} Average G_loss: {sum(g_losses)/len(g_losses):.4f} '
              f'Average D_loss: {sum(d_losses)/len(d_losses):.4f}')
        
        # Save models
        if (epoch + 1) % 10 == 0:
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict()
            }, f'outputs/gan_epoch_{epoch+1}.pth')
    
    return sum(g_losses)/len(g_losses), sum(d_losses)/len(d_losses) 