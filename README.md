# Implemented VAE and GAN on mango leaf disease dataset

## Installation

1. Clone the repository:
```bash
git clone https://github.com/himaenshuu/GAN_and_vae.git
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset
- Dataset: [https://www.kaggle.com/datasets/warcoder/mango-leaf-disease-dataset]

## Usage

1. Training the models:
```bash
python train.py
```

The script will:
- Train both VAE and GAN models
- Save model checkpoints
- Generate synthetic images
- Calculate evaluation metrics (FID, SSIM, Inception Score)

2. Model Parameters:
- VAE: Latent dimension = 100
- GAN: Noise dimension = 100
- Training epochs = 500
- Batch size = 32
- Learning rate = 0.0002

## Evaluation Metrics

The project implements three evaluation metrics:

1. **Fréchet Inception Distance (FID)**
   - Measures the distance between feature distributions of real and generated images
   - Lower scores indicate better quality

2. **Structural Similarity Index (SSIM)**
   - Measures the structural similarity between real and generated images
   - Higher scores indicate better quality

3. **Inception Score**
   - Measures the diversity and quality of generated images
   - Higher scores indicate better quality

## Results

The trained models will generate synthetic images that can be used for:
- Data augmentation
- Training other models
- Visual analysis of disease patterns

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

