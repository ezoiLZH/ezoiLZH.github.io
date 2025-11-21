import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from PIL import Image
import os
import cv2

# 解决OpenMP库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from nerf_components import RaysDataset, sample_along_rays, NeRF, volrend, psnr


def load_my_data(data_path):
    """
    Load the my dataset
    
    Args:
        data_path: Path to the my_nerf_data.npz file
        
    Returns:
        images_train, c2ws_train, images_val, c2ws_val, c2ws_test, focal
    """
    if not os.path.exists(data_path):
        # Try alternative path
        data_path = "my_nerf_data.npz"
        
    data = np.load(data_path)
    
    # Training images: [100, 200, 200, 3]
    images_train = data["images_train"] / 255.0
    
    # Cameras for the training images (camera-to-world transformation matrix): [100, 4, 4]
    c2ws_train = data["c2ws_train"]
    
    # Validation images: [10, 200, 200, 3]
    images_val = data["images_val"] / 255.0
    
    # Cameras for the validation images: [10, 4, 4]
    c2ws_val = data["c2ws_val"]
    
    # Test cameras for novel-view video rendering: [60, 4, 4]
    c2ws_test = data["c2ws_test"]
    
    # Camera focal length
    focal = data["focal"]  # float

    # Resize images to 1/16 of their original size
    images_train = resize_images(images_train, factor=0.25)  # 1/4 size in each dimension = 1/16 area
    images_val = resize_images(images_val, factor=0.25)      # 1/4 size in each dimension = 1/16 area
    
    # Adjust focal length according to the resize factor
    focal = focal * 0.25
    
    return images_train, c2ws_train, images_val, c2ws_val, c2ws_test, focal

def resize_images(images, factor=0.25):
    """
    Resize images by a given factor
    
    Args:
        images: Array of images with shape [N, H, W, 3]
        factor: Resize factor (0.25 means 1/4 size in each dimension)
        
    Returns:
        Resized images with shape [N, H*factor, W*factor, 3]
    """
    from PIL import Image
    resized_images = []
    
    for img in images:
        # Convert to PIL Image
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        
        # Calculate new dimensions
        new_width = int(img.shape[1] * factor)
        new_height = int(img.shape[0] * factor)
        
        # Resize image
        resized_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert back to numpy array and normalize
        resized_array = np.array(resized_img).astype(np.float32) / 255.0
        resized_images.append(resized_array)
    
    return np.array(resized_images)


def create_intrinsic_matrix(focal, H, W):
    """
    Create the intrinsic matrix K
    
    Args:
        focal: Focal length
        H: Image height
        W: Image width
        
    Returns:
        K: Intrinsic matrix of shape [3, 3]
    """
    K = np.array([
        [focal, 0, W/2],
        [0, focal, H/2],
        [0, 0, 1]
    ], dtype=np.float32)
    
    return K


def render_image(model, c2w, K, H, W, near=2.0, far=6.0, N_samples=64, device='cpu'):
    """
    Render a full image using the trained NeRF model
    
    Args:
        model: Trained NeRF model
        c2w: Camera-to-world transformation matrix
        K: Intrinsic matrix
        H: Image height
        W: Image width
        near: Near plane
        far: Far plane
        N_samples: Number of samples per ray
        device: Device to use
        
    Returns:
        Rendered image of shape [H, W, 3]
    """
    # Convert to tensors
    c2w = torch.tensor(c2w, dtype=torch.float32, device=device)
    K = torch.tensor(K, dtype=torch.float32, device=device)
    
    # Create pixel coordinate grid
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W, device=device), 
        torch.linspace(0, H-1, H, device=device), 
        indexing='ij'
    )
    i = i.T
    j = j.T
    uv = torch.stack([i, j], dim=-1).reshape(-1, 2)
    
    # Add 0.5 offset to move from pixel corner to pixel center
    uv = uv + 0.5
    
    # Expand dimensions for single camera
    uv = uv.unsqueeze(0)  # Shape: [1, H*W, 2]
    
    # Get rays
    # Ray origin is the camera center in world coordinates
    ray_o = c2w[:3, 3].unsqueeze(0).unsqueeze(1)  # Shape: [1, 1, 3]
    ray_o = ray_o.expand(-1, uv.shape[1], -1)  # Shape: [1, H*W, 3]
    
    # To calculate ray direction, we find a point along the ray at depth=1
    s = torch.ones_like(uv[..., 0])  # Shape: [1, H*W]
    
    # Convert pixel to camera coordinates
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    fx = fx.unsqueeze(0).unsqueeze(1).expand(-1, uv.shape[1])
    fy = fy.unsqueeze(0).unsqueeze(1).expand(-1, uv.shape[1])
    cx = cx.unsqueeze(0).unsqueeze(1).expand(-1, uv.shape[1])
    cy = cy.unsqueeze(0).unsqueeze(1).expand(-1, uv.shape[1])
    
    u = uv[..., 0]
    v = uv[..., 1]
    
    x = (u - cx) * s / fx
    y = (v - cy) * s / fy
    z = s
    
    x_c = torch.stack([x, y, z], dim=-1)  # Shape: [1, H*W, 3]
    
    # Transform camera coordinates to world coordinates
    ones = torch.ones_like(x_c[..., :1])
    x_c_hom = torch.cat([x_c, ones], dim=-1)
    x_w_hom = torch.matmul(c2w.unsqueeze(0), x_c_hom.unsqueeze(-1)).squeeze(-1)
    x_w = x_w_hom[..., :3]
    
    # Ray direction is from camera center to this point
    ray_d = x_w - ray_o
    ray_d = torch.nn.functional.normalize(ray_d, dim=-1)
    
    # Reshape rays
    ray_o = ray_o.reshape(-1, 3)
    ray_d = ray_d.reshape(-1, 3)
    
    # Sample points along rays
    step_size = (far - near) / N_samples
    points = sample_along_rays(ray_o, ray_d, near=near, far=far, N_samples=N_samples, perturb=False)
    
    # Reshape for model input
    batch_size = points.shape[0]
    points_flat = points.view(-1, 3)
    rays_d_flat = ray_d.unsqueeze(1).expand(-1, N_samples, -1).reshape(-1, 3)
    
    # Process in chunks to avoid memory issues
    chunk_size = 1024
    rgb_chunks = []
    sigma_chunks = []
    
    for i in range(0, points_flat.shape[0], chunk_size):
        p_chunk = points_flat[i:i+chunk_size]
        d_chunk = rays_d_flat[i:i+chunk_size]
        
        rgb_chunk, sigma_chunk = model(p_chunk, d_chunk)
        rgb_chunks.append(rgb_chunk)
        sigma_chunks.append(sigma_chunk)
    
    # Concatenate chunks
    rgb = torch.cat(rgb_chunks, dim=0)
    sigma = torch.cat(sigma_chunks, dim=0)
    
    # Reshape outputs
    rgb = rgb.view(batch_size, N_samples, 3)
    sigma = sigma.view(batch_size, N_samples, 1)
    
    # Volume rendering
    rendered_colors = volrend(sigma, rgb, step_size)
    
    # Reshape to image
    image = rendered_colors.view(H, W, 3)
    
    return image


def train_nerf_with_visualization():
    """
    Train the NeRF model and generate all required deliverables
    """
    print("=== NeRF Training for my Dataset ===")
    
    # Implementation description
    print("\nImplementation Description:")
    print("Part 2.1 - Ray Generation:")
    print("  - transform(c2w, x_c): Transforms points from camera to world coordinates")
    print("  - pixel_to_camera(K, uv, s): Converts pixel coordinates to camera coordinates")
    print("  - pixel_to_ray(K, c2w, uv): Converts pixel coordinates to rays")
    print("  - get_rays(H, W, K, c2w): Generates rays for a full image")
    
    print("\nPart 2.2 - Sampling:")
    print("  - RaysDataset: Precomputes all rays from multi-view images")
    print("  - sample_along_rays: Samples points along rays with optional perturbation")
    
    print("\nPart 2.4 - NeRF Network:")
    print("  - PositionalEncoding: Implements sinusoidal positional encoding")
    print("  - NeRF: 8-layer MLP with skip connections and view-dependent rendering")
    
    print("\nPart 2.5 - Volume Rendering:")
    print("  - volrend: Implements volume rendering equation with cumulative products")
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    print("\nLoading my dataset...")
    data_path = "my_nerf_data.npz"
    images_train, c2ws_train, images_val, c2ws_val, c2ws_test, focal = load_my_data(data_path)
    
    H, W = images_train.shape[1:3]
    K = create_intrinsic_matrix(focal, H, W)
    
    print(f"Data loaded:")
    print(f"  Training images: {images_train.shape}")
    print(f"  Training cameras: {c2ws_train.shape}")
    print(f"  Validation images: {images_val.shape}")
    print(f"  Validation cameras: {c2ws_val.shape}")
    print(f"  Test cameras: {c2ws_test.shape}")
    print(f"  Focal length: {focal}")
    print(f"  Image size: {H} x {W}")
    
    # Create dataset
    print("\nCreating rays dataset...")
    train_dataset = RaysDataset(images_train, K, c2ws_train, device=device)
    val_dataset = RaysDataset(images_val, K, c2ws_val, device=device)
    
    # Create model
    print("\nCreating NeRF model...")
    model = NeRF(coord_frequencies=10, dir_frequencies=4, hidden_channels=256, num_layers=8)
    model = model.to(device)
    
    # Check if a pre-trained model exists and load it
    model_path = "nerf_my_model.pth"
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Pre-trained model loaded successfully!")
        except Exception as e:
            print(f"Failed to load pre-trained model: {e}")
            print("Training from scratch...")
    else:
        print("No pre-trained model found. Training from scratch.")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    
    # Training parameters
    N_rays = 10000  # Batch size in terms of rays
    N_samples = 64  # Number of samples per ray
    near = 0.005
    far = 0.5
    step_size = (far - near) / N_samples
    N_epochs = 3000
    
    # For visualization and metrics
    psnr_history = []
    val_psnr_history = []
    loss_history = []
    
    # Save images at specific intervals
    save_epochs = [10, 100, 300, 500, 1000, 1500, 2000, 2500, 3000]
    rendered_images = {}
    
    print("\nStarting training...")

    # Save rendered images
    print("\nSaving rendered images...")
    os.makedirs('rendered_images', exist_ok=True)
    print("Rendered images saved to rendered_images/ directory")
    
    # Training loop
    for epoch in tqdm(range(N_epochs)):
        # Sample rays
        rays_o, rays_d, pixels = train_dataset.sample_rays(N_rays)
        
        # Sample points along rays
        points = sample_along_rays(rays_o, rays_d, near=near, far=far, N_samples=N_samples, perturb=True)
        
        # Reshape rays for model input
        batch_size = points.shape[0]
        points_flat = points.view(-1, 3)  # Shape: (batch_size * N_samples, 3)
        rays_d_flat = rays_d.unsqueeze(1).expand(-1, N_samples, -1).reshape(-1, 3)  # Shape: (batch_size * N_samples, 3)
        
        # Forward pass
        rgb, sigma = model(points_flat, rays_d_flat)
        
        # Reshape outputs
        rgb = rgb.view(batch_size, N_samples, 3)
        sigma = sigma.view(batch_size, N_samples, 1)
        
        # Volume rendering
        rendered_colors = volrend(sigma, rgb, step_size)
        
        # Calculate loss
        loss = torch.mean((rendered_colors - pixels)**2)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate PSNR
        psnr_value = psnr(loss)
        
        # Store metrics
        psnr_history.append(psnr_value.item())
        loss_history.append(loss.item())
        
        # Validation every 50 epochs
        if epoch % 50 == 0:
            with torch.no_grad():
                # Sample validation rays
                val_rays_o, val_rays_d, val_pixels = val_dataset.sample_rays(N_rays)
                
                # Sample points along rays
                val_points = sample_along_rays(val_rays_o, val_rays_d, near=near, far=far, N_samples=N_samples, perturb=False)
                
                # Reshape rays for model input
                val_batch_size = val_points.shape[0]
                val_points_flat = val_points.view(-1, 3)
                val_rays_d_flat = val_rays_d.unsqueeze(1).expand(-1, N_samples, -1).reshape(-1, 3)
                
                # Forward pass
                val_rgb, val_sigma = model(val_points_flat, val_rays_d_flat)
                
                # Reshape outputs
                val_rgb = val_rgb.view(val_batch_size, N_samples, 3)
                val_sigma = val_sigma.view(val_batch_size, N_samples, 1)
                
                # Volume rendering
                val_rendered_colors = volrend(val_sigma, val_rgb, step_size)
                
                # Calculate validation loss and PSNR
                val_loss = torch.mean((val_rendered_colors - val_pixels)**2)
                val_psnr = psnr(val_loss)
                
                val_psnr_history.append(val_psnr.item())
                
                # Print progress
                print(f"\nEpoch {epoch}: Train Loss = {loss.item():.6f}, Train PSNR = {psnr_value.item():.2f} dB, Val PSNR = {val_psnr.item():.2f} dB")

                # Plot training history
                print("\nGenerating training visualization...")
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.plot(loss_history)
                plt.title('Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.grid(True)
                
                plt.subplot(1, 3, 2)
                plt.plot(psnr_history, label='Training')
                plt.title('PSNR')
                plt.xlabel('Epoch')
                plt.ylabel('PSNR (dB)')
                plt.grid(True)
                plt.legend()
                
                # Plot validation PSNR at the right intervals
                val_epochs = list(range(0, N_epochs, 50))[:len(val_psnr_history)]
                plt.subplot(1, 3, 3)
                plt.plot(val_epochs, val_psnr_history, 'r-', label='Validation')
                plt.title('Validation PSNR')
                plt.xlabel('Epoch')
                plt.ylabel('PSNR (dB)')
                plt.grid(True)
                plt.legend()
                
                plt.tight_layout()
                plt.savefig('my_training_history.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("Training history saved to my_training_history.png")

                # Save model
                torch.save(model.state_dict(), "nerf_my_model.pth")
                print("Model saved to nerf_my_model.pth")

        # Save rendered images at specific epochs
        if epoch in save_epochs or epoch == N_epochs - 1:
            print(f"\nRendering validation image at epoch {epoch}...")
            with torch.no_grad():
                # Render one validation image
                val_image = render_image(model, c2ws_val[0], K, H, W, near, far, N_samples, device)
                rendered_images[epoch] = val_image.cpu().numpy()

                image_uint8 = (np.clip(rendered_images[epoch], 0, 1) * 255).astype(np.uint8)
                image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                img = Image.fromarray(image_uint8)
                img.save(f'rendered_images/epoch_{epoch}.png')
    
    print(f"\nFinal training results:")
    print(f"  Final Train Loss: {loss.item():.6f}")
    print(f"  Final Train PSNR: {psnr_value.item():.2f} dB")
    print(f"  Final Val PSNR: {val_psnr.item():.2f} dB")
    
    # Generate spherical rendering video
    print("\nGenerating spherical rendering video...")
    os.makedirs('spherical_rendering', exist_ok=True)
    
    # Render images from test cameras
    rendered_frames = []
    with torch.no_grad():
        for i, c2w in enumerate(tqdm(c2ws_test)):
            rendered_image = render_image(model, c2w, K, H, W, near, far, N_samples, device)
            image_np = rendered_image.cpu().numpy()
            image_uint8 = (np.clip(image_np, 0, 1) * 255).astype(np.uint8)
            image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            rendered_frames.append(image_uint8)
            
            # Save individual frames
            img = Image.fromarray(image_uint8)
            img.save(f'spherical_rendering/frame_{i:03d}.png')
    
    print("Spherical rendering frames saved to spherical_rendering/ directory")
    
    # Create a GIF from the frames
    try:
        frames_for_gif = [Image.fromarray(frame) for frame in rendered_frames]
        frames_for_gif[0].save(
            'spherical_rendering.gif',
            save_all=True,
            append_images=frames_for_gif[1:],
            duration=500,  # milliseconds per frame
            loop=0
        )
        print("Spherical rendering GIF saved to spherical_rendering.gif")
    except Exception as e:
        print(f"Failed to create GIF: {e}")
    
    # Save visualization data for rays and samples
    print("\nGenerating rays and samples visualization...")
    rays_o, rays_d, pixels = train_dataset.sample_rays(100)  # Sample only 100 rays
    points = sample_along_rays(rays_o, rays_d, near=near, far=far, N_samples=32, perturb=True)
    
    vis_data = {
        'rays_o': rays_o.cpu().numpy(),
        'rays_d': rays_d.cpu().numpy(),
        'points': points.cpu().numpy(),
        'pixels': pixels.cpu().numpy(),
        'cameras': c2ws_train,
        'K': K,
        'H': H,
        'W': W
    }
    
    np.save('rays_visualization_data.npy', vis_data)
    print("Rays and samples visualization data saved to rays_visualization_data.npy")
    
    print("\n=== All deliverables generated successfully ===")
    print("1. Implementation description included above")
    print("2. Rays and samples visualization data: rays_visualization_data.npy")
    print("3. Training progression images: rendered_images/ directory")
    print("4. Training history plot: training_history.png")
    print("5. Validation PSNR curve included in training_history.png")
    print("6. Spherical rendering video frames: spherical_rendering/ directory")
    print("7. Spherical rendering GIF: spherical_rendering.gif")
    print("8. Trained model: nerf_my_model.pth")


if __name__ == "__main__":
    train_nerf_with_visualization()