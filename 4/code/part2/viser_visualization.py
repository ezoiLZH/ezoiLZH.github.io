import viser
import time
import numpy as np
import torch
import os

from nerf_components import RaysDataset, sample_along_rays


def load_lego_data(data_path):
    """
    Load the Lego dataset
    
    Args:
        data_path: Path to the lego_200x200.npz file
        
    Returns:
        images_train, c2ws_train, images_val, c2ws_val, c2ws_test, focal
    """
    if not os.path.exists(data_path):
        # Try alternative path
        data_path = "lego_200x200.npz"
        
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
    
    return images_train, c2ws_train, images_val, c2ws_val, c2ws_test, focal


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


def visualize_with_viser():
    """
    Visualize cameras, rays and samples using viser as required
    """
    print("Loading Lego data...")
    
    # Load data
    data_path = "lego_200x200.npz"
    images_train, c2ws_train, images_val, c2ws_val, c2ws_test, focal = load_lego_data(data_path)
    
    H, W = images_train.shape[1:3]
    K = create_intrinsic_matrix(focal, H, W)
    
    print(f"Data loaded:")
    print(f"  Training images: {images_train.shape}")
    print(f"  Training cameras: {c2ws_train.shape}")
    print(f"  Focal length: {focal}")
    print(f"  Image size: {H} x {W}")
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    print("Creating rays dataset...")
    dataset = RaysDataset(images_train, K, c2ws_train, device=device)
    
    # --- You Need to Implement These ------
    rays_o, rays_d, pixels = dataset.sample_rays(100) # Should expect (B, 3)
    points = sample_along_rays(rays_o, rays_d, perturb=True)
    # ---------------------------------------
    
    print(f"Visualization data:")
    print(f"  Rays origin: {rays_o.shape}")
    print(f"  Rays direction: {rays_d.shape}")
    print(f"  Sampled points: {points.shape}")
    
    # Convert to numpy for visualization
    rays_o_np = rays_o.cpu().numpy()
    rays_d_np = rays_d.cpu().numpy()
    points_np = points.cpu().numpy()
    
    # Save visualization data
    vis_data = {
        'rays_o': rays_o_np,
        'rays_d': rays_d_np,
        'points': points_np,
        'pixels': pixels.cpu().numpy(),
        'cameras': c2ws_train,
        'K': K,
        'H': H,
        'W': W
    }
    
    np.save('rays_visualization_data.npy', vis_data)
    print("Visualization data saved to rays_visualization_data.npy")
    
    # Start viser server
    print("Starting viser server...")
    server = viser.ViserServer(share=True)
    
    # Add cameras
    for i, (image, c2w) in enumerate(zip(images_train, c2ws_train)):
        server.add_camera_frustum(
            f"/cameras/{i}",
            fov=2 * np.arctan2(H / 2, K[0, 0]),
            aspect=W / H,
            scale=0.15,
            wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz,
            position=c2w[:3, 3],
            image=image
        )
    
    # Add rays
    for i, (o, d) in enumerate(zip(rays_o_np, rays_d_np)):
        server.add_spline_catmull_rom(
            f"/rays/{i}", positions=np.stack((o, o + d * 6.0)),
        )
    
    # Add sample points
    server.add_point_cloud(
        f"/samples",
        colors=np.zeros_like(points_np).reshape(-1, 3),
        points=points_np.reshape(-1, 3),
        point_size=0.02,
    )
    
    print("Visualization is ready. Connect to the viser server to view the visualization.")
    print("The server is running. Press Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(0.1)  # Wait to allow visualization to run
    except KeyboardInterrupt:
        print("\nStopping viser server...")
        server.stop()
        print("Viser server stopped.")


def verify_pixel_coordinates():
    """
    Verify that pixel coordinates are correctly mapped
    """
    print("Loading Lego data for verification...")
    
    # Load data
    data_path = "lego_200x200.npz"
    images_train, c2ws_train, images_val, c2ws_val, c2ws_test, focal = load_lego_data(data_path)
    
    H, W = images_train.shape[1:3]
    K = create_intrinsic_matrix(focal, H, W)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataset
    dataset = RaysDataset(images_train, K, c2ws_train, device=device)
    
    # This will check that your uvs aren't flipped
    uvs_start = 0
    uvs_end = min(40_000, len(dataset.uvs))
    sample_uvs = dataset.uvs[uvs_start:uvs_end] # These are integer coordinates of widths / heights (xy not yx) of all the pixels in an image
    # uvs are array of xy coordinates, so we need to index into the 0th image tensor with [0, height, width], so we need to index with uv[:,1] and then uv[:,0]
    
    try:
        pixel_values = images_train[0, sample_uvs[:, 1], sample_uvs[:, 0]]
        dataset_pixels = dataset.pixels[uvs_start:uvs_end].cpu().numpy()
        
        # Check if they match
        match = np.allclose(pixel_values, dataset_pixels, atol=1e-6)
        print(f"Pixel coordinate mapping verification: {'PASSED' if match else 'FAILED'}")
        
        if not match:
            print("  First 5 expected values:", pixel_values[:5])
            print("  First 5 dataset values:", dataset_pixels[:5])
    except Exception as e:
        print(f"Pixel coordinate mapping verification failed with error: {e}")


if __name__ == "__main__":
    verify_pixel_coordinates()
    visualize_with_viser()