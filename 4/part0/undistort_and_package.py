import cv2
import numpy as np
import glob
import os
from estimate_pose import estimate_camera_poses, load_calibration_params

def undistort_images_and_poses(images_path, output_path="my_data.npz", tag_size=0.02, 
                              train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Load camera calibration parameters
    camera_matrix, dist_coeffs = load_calibration_params()
    print("Camera calibration parameters loaded")
    
    # Estimate camera poses
    print("Estimating camera poses...")
    poses, images = estimate_camera_poses(
        images_path=images_path,
        tag_size=tag_size,
        calibration_file="camera_params.npz"
    )
    
    print(f"Processing {len(images)} images...")
    
    # Lists to store undistorted images and poses
    undistorted_images = []
    c2ws = []  # Camera-to-world matrices
    
    # Process each image
    for i, (image_file, c2w) in enumerate(zip(images, poses)):
        print(f"Processing image {i+1}/{len(images)}: {os.path.basename(image_file)}")
        
        # Read image
        img = cv2.imread(image_file)
        if img is None:
            print(f"Warning: Could not read image {image_file}")
            continue
            
        # Get image dimensions
        h, w = img.shape[:2]
        
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), alpha=0, newImgSize=(w, h)
        )
        undistorted_img = cv2.undistort(
            img, camera_matrix, dist_coeffs, None, new_camera_matrix
        )
        
        # Crop to the valid region of interest
        x, y, w_roi, h_roi = roi
        undistorted_img = undistorted_img[y:y+h_roi, x:x+w_roi]
        
        # Update the principal point to account for the crop offset
        new_camera_matrix[0, 2] -= x  # cx
        new_camera_matrix[1, 2] -= y  # cy
        
        # Store results
        undistorted_images.append(undistorted_img)
        c2ws.append(c2w)
    
    # Convert to numpy arrays
    c2ws = np.array(c2ws)
    
    # Split into train/val/test sets
    n_total = len(undistorted_images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    
    # Shuffle indices
    indices = np.random.permutation(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train+n_val]
    test_indices = indices[n_train+n_val:]
    
    print(f"Dataset split - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    # Prepare training data
    images_train = np.array([undistorted_images[i] for i in train_indices])
    c2ws_train = c2ws[train_indices]
    
    # Prepare validation data
    images_val = np.array([undistorted_images[i] for i in val_indices])
    c2ws_val = c2ws[val_indices]
    
    # Prepare test data (only poses, no images)
    c2ws_test = c2ws[test_indices]
    
    # Extract focal length (assuming fx = fy)
    focal = new_camera_matrix[0, 0]
    
    # Save the dataset
    np.savez(
        output_path,
        images_train=images_train,    # (N_train, H, W, 3)
        c2ws_train=c2ws_train,        # (N_train, 4, 4)
        images_val=images_val,        # (N_val, H, W, 3)
        c2ws_val=c2ws_val,            # (N_val, 4, 4)
        c2ws_test=c2ws_test,          # (N_test, 4, 4)
        focal=focal                   # float
    )
    
    print(f"Dataset saved to {output_path}")
    print("Dataset contents:")
    print(f"  images_train: {images_train.shape}")
    print(f"  c2ws_train: {c2ws_train.shape}")
    print(f"  images_val: {images_val.shape}")
    print(f"  c2ws_val: {c2ws_val.shape}")
    print(f"  c2ws_test: {c2ws_test.shape}")
    print(f"  focal: {focal}")
    
    return output_path

def load_nerf_data(filepath):
    data = np.load(filepath)
    return {
        'images_train': data['images_train'],
        'c2ws_train': data['c2ws_train'],
        'images_val': data['images_val'],
        'c2ws_val': data['c2ws_val'],
        'c2ws_test': data['c2ws_test'],
        'focal': data['focal']
    }

def visualize_dataset_samples(filepath, num_samples=3):
    try:
        import matplotlib.pyplot as plt
        
        data = load_nerf_data(filepath)
        images_train = data['images_train']
        
        fig, axes = plt.subplots(1, min(num_samples, len(images_train)), figsize=(15, 5))
        if num_samples == 1:
            axes = [axes]
            
        for i in range(min(num_samples, len(images_train))):
            # Convert BGR to RGB for proper display
            img_rgb = cv2.cvtColor(images_train[i], cv2.COLOR_BGR2RGB)
            axes[i].imshow(img_rgb)
            axes[i].set_title(f'Sample {i+1}')
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for visualization")

def main():
    images_path = "object_scan_images"
    if not os.path.exists(images_path):
        print(f"Directory '{images_path}' does not exist.")
        print("Please create this directory and place your object scan images inside.")
        return
    
    image_files = [f for f in os.listdir(images_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No images found in '{images_path}' directory.")
        print("Please add your object scan images to this directory.")
        return
    
    print(f"Found {len(image_files)} images for processing")
    
    # Process images and create dataset
    try:
        print("Starting image undistortion and dataset packaging...")
        dataset_path = undistort_images_and_poses(
            images_path=images_path,
            output_path="my_nerf_data.npz",
            tag_size=0.0535,  # 2cm tags - adjust to match your printed tags
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1
        )
        
        print(f"\nDataset successfully created at: {dataset_path}")
        
        # Load and inspect the dataset
        print("\nLoading dataset for inspection...")
        data = load_nerf_data(dataset_path)
        
        print("Dataset inspection:")
        print(f"  Training images: {data['images_train'].shape}")
        print(f"  Training poses: {data['c2ws_train'].shape}")
        print(f"  Validation images: {data['images_val'].shape}")
        print(f"  Validation poses: {data['c2ws_val'].shape}")
        print(f"  Test poses: {data['c2ws_test'].shape}")
        print(f"  Focal length: {data['focal']:.2f}")
        
        # Try to visualize sample images
        print("\nDisplaying sample images...")
        visualize_dataset_samples(dataset_path, num_samples=3)
        
    except Exception as e:
        print(f"Processing failed: {e}")

if __name__ == "__main__":
    main()