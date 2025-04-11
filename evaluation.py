import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os
from PIL import Image
import torchvision.transforms as transforms

def calculate_ssim(real_images, generated_images):
    """
    Calculate SSIM between real and generated images
    """
    # Convert to numpy and ensure same shape
    real_images = real_images.cpu().numpy()
    generated_images = generated_images.cpu().numpy()
    
    # Print shapes for debugging
    print(f"Real images shape: {real_images.shape}")
    print(f"Generated images shape: {generated_images.shape}")
    
    # Ensure same number of images
    num_images = min(len(real_images), len(generated_images))
    real_images = real_images[:num_images]
    generated_images = generated_images[:num_images]
    
    ssim_scores = []
    for i in range(num_images):
        # Get image dimensions
        h, w = real_images[i].shape[1:]
        # Use a smaller window size if image is small
        win_size = min(7, h, w)
        if win_size % 2 == 0:
            win_size -= 1  # Ensure window size is odd
            
        # Convert to channel last format and ensure float32
        real_img = real_images[i].transpose(1, 2, 0).astype(np.float32)
        gen_img = generated_images[i].transpose(1, 2, 0).astype(np.float32)
        
        # Ensure images have same shape
        if real_img.shape != gen_img.shape:
            print(f"Shape mismatch at index {i}: real={real_img.shape}, gen={gen_img.shape}")
            continue
            
        # Calculate SSIM with proper channel handling and data range
        score = ssim(real_img, gen_img,
                    multichannel=True,
                    win_size=win_size,
                    channel_axis=2,
                    data_range=1.0)  # Since our images are normalized to [0,1]
        ssim_scores.append(score)
    
    if not ssim_scores:
        print("Warning: No valid SSIM scores calculated")
        return 0.0
        
    return np.mean(ssim_scores)

def save_images_for_fid(images, output_dir, prefix):
    """
    Save images for FID calculation
    """
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(images):
        # Convert to PIL Image
        img = transforms.ToPILImage()(img.cpu())
        # Save as PNG
        img.save(os.path.join(output_dir, f'{prefix}_{i}.png'))

def generate_and_save_images(model, num_images, output_path, model_type='vae', device=None):
    """
    Generate and save images from the model
    """
    # Get device from model if not provided
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    with torch.no_grad():
        if model_type == 'vae':
            # For VAE, generate random latent vectors
            z = torch.randn(num_images, 128, device=device)
            generated_images = model.decoder(z)
        else:
            # For GAN, generate random noise
            z = torch.randn(num_images, 100, 1, 1, device=device)
            generated_images = model(z)
            # Denormalize GAN output
            generated_images = (generated_images + 1) / 2
        
        # Create grid of images
        grid = make_grid(generated_images, nrow=10, normalize=True)
        
        # Save grid
        plt.figure(figsize=(20, 20))
        plt.imshow(grid.permute(1, 2, 0).cpu())
        plt.axis('off')
        plt.title(f'{model_type.upper()} Generated Samples')
        plt.savefig(output_path)
        plt.close()
        
        return generated_images

def evaluate_models(vae, gan, val_loader, device=None, output_dir='outputs'):
    """
    Evaluate both models and save results
    """
    # Get device from models if not provided
    if device is None:
        device = next(vae.parameters()).device
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    real_dir = os.path.join(output_dir, 'real_images')
    vae_dir = os.path.join(output_dir, 'vae_images')
    gan_dir = os.path.join(output_dir, 'gan_images')
    
    # Get a batch of real images
    real_images, _ = next(iter(val_loader))
    real_images = real_images.to(device)
    
    # Generate images from both models
    vae_images = generate_and_save_images(vae, len(real_images), 
                                        os.path.join(output_dir, 'vae_samples.png'), 
                                        'vae', device)
    gan_images = generate_and_save_images(gan, len(real_images), 
                                        os.path.join(output_dir, 'gan_samples.png'), 
                                        'gan', device)
    
    # Calculate SSIM scores
    vae_ssim = calculate_ssim(real_images, vae_images)
    gan_ssim = calculate_ssim(real_images, gan_images)
    
    # Save images for FID calculation
    save_images_for_fid(real_images, real_dir, 'real')
    save_images_for_fid(vae_images, vae_dir, 'vae')
    save_images_for_fid(gan_images, gan_dir, 'gan')
    
    # Print results
    print(f'VAE SSIM: {vae_ssim:.4f}')
    print(f'GAN SSIM: {gan_ssim:.4f}')
    
    return {
        'vae_ssim': vae_ssim,
        'gan_ssim': gan_ssim,
        'real_images': real_images,
        'vae_images': vae_images,
        'gan_images': gan_images
    } 