import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from data_utils import MangoLeafDataset
from vae import VAE
from gan import Generator, Discriminator
from evaluation import evaluate_models
import matplotlib.pyplot as plt

def train_vae_epoch(model, train_loader, optimizer, device, epoch):
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
        
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\\tLoss: {loss.item() / len(data):.6f}')
    
    return train_loss / len(train_loader.dataset)

def train_gan_epoch(generator, discriminator, train_loader, g_optimizer, d_optimizer, device, epoch):
    generator.train()
    discriminator.train()
    
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
        d_loss_real = nn.functional.binary_cross_entropy(output_real, label_real)
        
        # Fake images
        noise = torch.randn(batch_size, 100, 1, 1).to(device)
        fake_imgs = generator(noise)
        output_fake = discriminator(fake_imgs.detach())
        label_fake = torch.zeros(batch_size).to(device)
        d_loss_fake = nn.functional.binary_cross_entropy(output_fake, label_fake)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()
        
        # Train Generator
        g_optimizer.zero_grad()
        output_fake = discriminator(fake_imgs)
        g_loss = nn.functional.binary_cross_entropy(output_fake, label_real)
        g_loss.backward()
        g_optimizer.step()
        
        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())
        
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(real_imgs)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\\t'
                  f'D_loss: {d_loss.item():.6f}\\tG_loss: {g_loss.item():.6f}')
    
    return sum(g_losses) / len(g_losses), sum(d_losses) / len(d_losses)

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = MangoLeafDataset(root_dir='.', transform=transform)
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Initialize models
    vae = VAE().to(device)
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # Initialize optimizers
    vae_optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    g_optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    
    # Training parameters
    num_epochs = 10
    
    # Lists to store losses
    vae_losses = []
    gan_g_losses = []
    gan_d_losses = []
    
    # Train VAE
    print("Training VAE...")
    for epoch in range(1, num_epochs + 1):
        train_loss = train_vae_epoch(vae, train_loader, vae_optimizer, device, epoch)
        vae_losses.append(train_loss)
        print(f'====> Epoch: {epoch} Average loss: {train_loss:.4f}')
        
        if epoch % 10 == 0:
            torch.save(vae.state_dict(), f'outputs/vae_epoch_{epoch}.pth')
    
    # Train GAN
    print("\nTraining GAN...")
    for epoch in range(1, num_epochs + 1):
        g_loss, d_loss = train_gan_epoch(generator, discriminator, train_loader, 
                                       g_optimizer, d_optimizer, device, epoch)
        gan_g_losses.append(g_loss)
        gan_d_losses.append(d_loss)
        print(f'====> Epoch: {epoch} G_loss: {g_loss:.4f} D_loss: {d_loss:.4f}')
        
        if epoch % 10 == 0:
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict()
            }, f'outputs/gan_epoch_{epoch}.pth')
    
    # Plot loss curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(vae_losses)
    plt.title('VAE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(gan_g_losses, label='Generator')
    plt.plot(gan_d_losses, label='Discriminator')
    plt.title('GAN Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('outputs/loss_curves.png')
    plt.close()
    
    # Evaluate models
    print("\nEvaluating models...")
    results = evaluate_models(vae, generator, val_loader, device, 'outputs')
    
    # Print final results
    print("\nFinal Results:")
    print(f"VAE SSIM: {results['vae_ssim']:.4f}")
    print(f"GAN SSIM: {results['gan_ssim']:.4f}")

if __name__ == "__main__":
    main() 
    
    