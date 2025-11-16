import torch
import torch.nn as nn
import numpy as np
import math
import os
import matplotlib.pyplot as plt

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding for neural fields
    """
    def __init__(self, num_frequencies=10):
        super(PositionalEncoding, self).__init__()
        self.num_frequencies = num_frequencies
        
    def forward(self, x):
        """
        Apply positional encoding to input coordinates
        
        Args:
            x: Input coordinates of shape (batch_size, 2)
            
        Returns:
            Encoded coordinates of shape (batch_size, 2 + 2*2*num_frequencies)
        """
        # Keep original coordinates
        encoded = [x]
        
        # Apply sin and cos for each frequency
        for i in range(self.num_frequencies):
            freq = 2 ** i
            encoded.append(torch.sin(freq * x))
            encoded.append(torch.cos(freq * x))
            
        # Concatenate all features
        return torch.cat(encoded, dim=-1)

class NeuralField(nn.Module):
    """
    Neural Field MLP with positional encoding
    """
    def __init__(self, num_frequencies=10, hidden_channels=256, num_layers=4):
        super(NeuralField, self).__init__()
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(num_frequencies)
        
        # Calculate input dimension after positional encoding
        # Original 2D coordinates + 2*2*num_frequencies (sin and cos for x and y)
        input_dim = 2 + 2 * 2 * num_frequencies
        
        # Build MLP layers
        layers = []
        layers.append(nn.Linear(input_dim, hidden_channels))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers-2):
            layers.append(nn.Linear(hidden_channels, hidden_channels))
            layers.append(nn.ReLU())
            
        # Output layer (3 channels for RGB)
        layers.append(nn.Linear(hidden_channels, 3))
        layers.append(nn.Sigmoid())  # Constrain output to [0, 1]
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, coords):
        """
        Forward pass through the neural field
        
        Args:
            coords: Input coordinates of shape (batch_size, 2)
            
        Returns:
            Predicted colors of shape (batch_size, 3)
        """
        # Apply positional encoding
        encoded_coords = self.pos_encoding(coords)
        
        # Pass through MLP
        colors = self.mlp(encoded_coords)
        
        return colors

class ImageDataset(torch.utils.data.Dataset):
    """
    Dataset for sampling pixels from an image
    """
    def __init__(self, image_path, device='cuda'):
        """
        Args:
            image_path: Path to the image file
            device: Device to store the data on
        """
        import cv2
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Store original image
        self.image = image
        
        # Normalize image to [0, 1]
        self.image_normalized = image.astype(np.float32) / 255.0
        
        # Get image dimensions
        self.height, self.width = image.shape[:2]
        
        # Create coordinate grid
        x_coords = np.arange(self.width)
        y_coords = np.arange(self.height)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        
        # Flatten coordinates and colors
        self.coords = np.stack([x_grid.flatten(), y_grid.flatten()], axis=-1)
        self.colors = self.image_normalized.reshape(-1, 3)
        
        # Convert to tensors
        self.coords = torch.tensor(self.coords, dtype=torch.float32, device=device)
        self.colors = torch.tensor(self.colors, dtype=torch.float32, device=device)
        
        # Normalize coordinates to [0, 1]
        self.coords[:, 0] = self.coords[:, 0] / self.width
        self.coords[:, 1] = self.coords[:, 1] / self.height
        
    def __len__(self):
        return len(self.coords)
    
    def __getitem__(self, idx):
        return self.coords[idx], self.colors[idx]

def psnr(mse):
    return -10 * torch.log10(mse + 1e-8)

def train_neural_field(image_path, 
                      num_frequencies=10, 
                      hidden_channels=256, 
                      num_layers=4,
                      batch_size=10000,
                      learning_rate=1e-2,
                      num_iterations=2000,
                      device='cuda',
                      save_progress=True):
    """
    Train a neural field to fit a 2D image
    
    Args:
        image_path: Path to the image file
        num_frequencies: Number of frequencies for positional encoding
        hidden_channels: Width of hidden layers
        num_layers: Number of layers in the MLP
        batch_size: Number of pixels to sample per iteration
        learning_rate: Learning rate for optimizer
        num_iterations: Number of training iterations
        device: Device to train on
        save_progress: Whether to save intermediate results
        
    Returns:
        Trained model and training history
    """
    # Create dataset
    dataset = ImageDataset(image_path, device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    model = NeuralField(
        num_frequencies=num_frequencies,
        hidden_channels=hidden_channels,
        num_layers=num_layers
    ).to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'loss': [],
        'psnr': []
    }
    
    # Training loop
    model.train()
    for iteration in range(num_iterations):
        coords_batch, colors_batch = next(iter(dataloader))
        
        pred_colors = model(coords_batch)
        
        loss = criterion(pred_colors, colors_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        psnr_value = psnr(loss)
        
        history['loss'].append(loss.item())
        history['psnr'].append(psnr_value.item())
        
        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration+1}/{num_iterations}, Loss: {loss.item():.6f}, PSNR: {psnr_value.item():.2f}")
            
        # Save intermediate results
        if save_progress and (iteration + 1) % 100 == 0:
            with torch.no_grad():
                model.eval()
                # Reconstruct full image
                reconstructed = model(dataset.coords)
                reconstructed_image = reconstructed.cpu().numpy().reshape(dataset.height, dataset.width, 3)
                # Save image
                import cv2
                reconstructed_image = (reconstructed_image * 255).astype(np.uint8)
                reconstructed_image = cv2.cvtColor(reconstructed_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"results_custom/reconstruction_iter_{iteration+1}.png", reconstructed_image)
                model.train()
    
    return model, history, dataset

def reconstruct_image(model, dataset, output_path="reconstruction.png"):
    """
    Reconstruct the full image using the trained model
    
    Args:
        model: Trained neural field model
        dataset: ImageDataset object
        output_path: Path to save the reconstructed image
    """
    model.eval()
    with torch.no_grad():
        # Reconstruct full image
        reconstructed = model(dataset.coords)
        reconstructed_image = reconstructed.cpu().numpy().reshape(dataset.height, dataset.width, 3)
        
        # Save image
        import cv2
        reconstructed_image = (reconstructed_image * 255).astype(np.uint8)
        reconstructed_image = cv2.cvtColor(reconstructed_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"results_custom/{output_path}", reconstructed_image)
        
    print(f"Reconstructed image saved to results_custom/{output_path}")

def plot_training_history(history, save_path="training_history.png"):
    """
    Plot training history (loss and PSNR)
    
    Args:
        history: Dictionary with 'loss' and 'psnr' keys
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(history['loss'])
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('MSE Loss')
    ax1.set_yscale('log')
    
    # Plot PSNR
    ax2.plot(history['psnr'])
    ax2.set_title('PSNR')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('PSNR (dB)')
    
    plt.tight_layout()
    plt.savefig(f"results_custom/{save_path}")
    plt.close()
    print(f"Training history plot saved to {save_path}")

def compare_hyperparameters(image_path, device='cpu'):
    """
    Compare different hyperparameter settings
    
    Args:
        image_path: Path to the image file
        device: Device to train on
    """
    # Hyperparameter settings to compare
    configs = [
        {"num_frequencies": 3, "hidden_channels": 64, "name": "Low Freq, Narrow"},
        {"num_frequencies": 3, "hidden_channels": 256, "name": "Low Freq, Wide"},
        {"num_frequencies": 10, "hidden_channels": 64, "name": "High Freq, Narrow"},
        {"num_frequencies": 10, "hidden_channels": 256, "name": "High Freq, Wide"}
    ]
    
    results = []
    
    for config in configs:
        print(f"\nTraining with config: {config['name']}")
        print(f"  Num frequencies: {config['num_frequencies']}")
        print(f"  Hidden channels: {config['hidden_channels']}")
        
        # Train model
        model, history, dataset = train_neural_field(
            image_path=image_path,
            num_frequencies=config['num_frequencies'],
            hidden_channels=config['hidden_channels'],
            num_layers=4,
            batch_size=10000,
            learning_rate=1e-2,
            num_iterations=1500,
            device=device,
            save_progress=False
        )
        
        # Save final result
        output_path = f"reconstruction_{config['name'].lower().replace(' ', '_').replace(',', '')}.png"
        reconstruct_image(model, dataset, output_path)
        
        # Store results
        results.append({
            "config": config,
            "history": history,
            "final_psnr": history['psnr'][-1],
            "final_loss": history['loss'][-1]
        })
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, result in enumerate(results):
        ax = axes[i]
        ax.plot(result['history']['psnr'])
        ax.set_title(f"{result['config']['name']}\nFinal PSNR: {result['final_psnr']:.2f} dB")
        ax.set_xlabel('Iteration')
        ax.set_ylabel('PSNR (dB)')
    
    plt.tight_layout()
    plt.savefig("hyperparameter_comparison.png")
    plt.close()
    print("Hyperparameter comparison saved to hyperparameter_comparison.png")
    
    return results

def main():
    # Check if PyTorch is available
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    else:
        print("CUDA available, using GPU")
        device = 'cuda'
    
    image_path = "custom.jpg"
    print(f"Using image: {image_path}")
    
    # Train a basic model
    print("\nTraining basic model...")
    model, history, dataset = train_neural_field(
        image_path=image_path,
        num_frequencies=10,
        hidden_channels=256,
        num_layers=4,
        batch_size=10000,
        learning_rate=1e-2,
        num_iterations=2000,
        device=device
    )
    
    # Save final reconstruction
    reconstruct_image(model, dataset, "final_reconstruction.png")
    
    # Plot training history
    plot_training_history(history, "training_history.png")
    
    # Compare hyperparameters
    # print("\nComparing hyperparameters...")
    # compare_hyperparameters(image_path, device)

if __name__ == "__main__":
    main()