import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm
import sys
import math
import cv2

# Add the part2.6 directory to the path so we can import nerf_components
sys.path.append(os.path.join(os.path.dirname(__file__)))

from nerf_components import NeRF, sample_along_rays, volrend
from complete_nerf_training import load_my_data, create_intrinsic_matrix, render_image


def generate_360_path(center=[0, 0, 0], radius=1.0, num_frames=120, elevation=0.1):
    """
    Generate camera poses for a 360-degree path around the object
    
    Args:
        center: Center point to orbit around
        radius: Distance from center
        num_frames: Number of frames in the video
        elevation: Elevation angle (in radians)
        
    Returns:
        c2ws: Camera-to-world transformation matrices [num_frames, 4, 4]
    """
    c2ws = []
    
    for i in range(num_frames):
        # Calculate angle for this frame
        angle = 2 * np.pi * i / num_frames
        
        # Camera position (x, y, z)
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        z = center[2] + radius * elevation

        c2w = np.array([[y, 0, -x, x],
                       [-x, 0, -y, y],
                       [0, 1, 0, z],
                       [0, 0, 0, 1]])
                       
        print(c2w)

        c2ws.append(c2w)
    
    return np.array(c2ws)


def load_trained_model(model_path, device):
    """
    Load a trained NeRF model
    
    Args:
        model_path: Path to the saved model
        device: Device to load the model on
        
    Returns:
        model: Loaded NeRF model
    """
    model = NeRF(coord_frequencies=10, dir_frequencies=4, hidden_channels=256, num_layers=8)
    model = model.to(device)
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully!")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model.eval()
    return model


def generate_360_video():
    """
    Generate a 360-degree video around the object using the trained NeRF model
    """
    print("=== Generating 360-Degree Video ===")
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data to get image dimensions and focal length
    print("Loading dataset information...")
    data_path = "my_nerf_data.npz"
    images_train, c2ws_train, images_val, c2ws_val, c2ws_test, focal = load_my_data(data_path)
    
    # Get image dimensions (use training image dimensions)
    H, W = images_train.shape[1:3]
    print(f"Image dimensions: {H} x {W}")
    print(f"Focal length: {focal}")
    
    # Create intrinsic matrix
    K = create_intrinsic_matrix(focal, H, W)
    
    # Load trained model
    model_path = "nerf_my_model.pth"
    model = load_trained_model(model_path, device)
    
    # Generate camera path for 360 video
    print("Generating camera path...")
    num_frames = 60
    c2ws_360 = generate_360_path(
        center=[0, 0, 0],      # Center around origin
        radius=1,            # Distance from center
        num_frames=num_frames, # Number of frames
        elevation=-0.1          # Slight elevation
    )
    
    # Create output directory
    output_dir = "360_video_frames"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving frames to: {output_dir}")
    
    # Render parameters
    near = 0.005
    far = 1
    N_samples = 64
    
    # Render frames
    print("Rendering frames...")
    rendered_frames = []
    
    for i in tqdm(range(num_frames)):
        c2w = c2ws_360[i]
        
        # Render image
        with torch.no_grad():
            rendered_image = render_image(
                model, c2w, K, H, W, 
                near=near, far=far, N_samples=N_samples, 
                device=device
            )
        
        # Convert to numpy and uint8
        image_np = rendered_image.cpu().numpy()
        image_uint8 = (np.clip(image_np, 0, 1) * 255).astype(np.uint8)
        image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        rendered_frames.append(image_uint8)
        
        # Save individual frame
        img = Image.fromarray(image_uint8)
        img.save(os.path.join(output_dir, f"frame_{i:03d}.png"))
    
    # Create GIF
    print("Creating GIF...")
    gif_path = "360_video.gif"
    
    try:
        frames_for_gif = [Image.fromarray(frame) for frame in rendered_frames]
        frames_for_gif[0].save(
            gif_path,
            save_all=True,
            append_images=frames_for_gif[1:],
            duration=100,  # milliseconds per frame
            loop=0
        )
        print(f"360-degree video saved to: {gif_path}")
    except Exception as e:
        print(f"Failed to create GIF: {e}")
    
    print("=== 360-Degree Video Generation Complete ===")


if __name__ == "__main__":
    generate_360_video()