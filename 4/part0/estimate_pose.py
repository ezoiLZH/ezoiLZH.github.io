import cv2
import numpy as np
import glob
import os
import time
import viser

def load_calibration_params(filename="camera_params.npz"):

    try:
        data = np.load(filename)
        camera_matrix = data['camera_matrix']
        dist_coeffs = data['dist_coeffs']
        print(f"Calibration parameters loaded from {filename}")
        return camera_matrix, dist_coeffs
    except FileNotFoundError:
        raise FileNotFoundError(f"Calibration file {filename} not found. Please run camera calibration first.")

def estimate_camera_poses(images_path, tag_size=0.02, calibration_file="camera_params.npz"):
    
    # Load camera calibration parameters
    camera_matrix, dist_coeffs = load_calibration_params(calibration_file)
    
    # Create ArUco dictionary and detector parameters (4x4 tags)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    
    # Define the real-world coordinates of the tag corners
    object_points = np.array([
        [0, 0, 0],           # Top-left corner
        [tag_size, 0, 0],    # Top-right corner
        [tag_size, tag_size, 0],  # Bottom-right corner
        [0, tag_size, 0]     # Bottom-left corner
    ], dtype=np.float32)
    
    # Get list of image files
    image_files = glob.glob(os.path.join(images_path, "*.jpg")) + \
                  glob.glob(os.path.join(images_path, "*.png")) + \
                  glob.glob(os.path.join(images_path, "*.jpeg"))
    
    print(f"Found {len(image_files)} images for pose estimation")
    
    # Lists to store results
    poses = []      # Camera-to-world transformation matrices
    images = []     # Filenames of images with detected tags
    
    # Process each image
    processed_images = 0
    for idx, image_file in enumerate(image_files):
        print(f"Processing image {idx+1}/{len(image_files)}: {os.path.basename(image_file)}")
        
        # Read image
        image = cv2.imread(image_file)
        if image is None:
            print(f"Warning: Could not read image {image_file}")
            continue
            
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers in the image
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
        
        # Check if any markers were detected
        if ids is not None and len(ids) > 0:
            print(f"  Detected {len(ids)} markers")
            
            # Use the first detected marker (assuming single tag per image)
            # Reshape corners to the format required by solvePnP
            image_points = corners[0][0].astype(np.float32)
            
            # Estimate pose using solvePnP
            success, rvec, tvec = cv2.solvePnP(
                object_points, 
                image_points, 
                camera_matrix, 
                dist_coeffs
            )
            
            if success:
                # Convert rotation vector to rotation matrix
                R, _ = cv2.Rodrigues(rvec)
                
                # Create world-to-camera transformation matrix (OpenCV format)
                w2c = np.eye(4)
                w2c[:3, :3] = R
                w2c[:3, 3] = tvec.flatten()
                
                # Convert to camera-to-world transformation matrix
                c2w = np.linalg.inv(w2c)
                
                # Store results
                poses.append(c2w)
                images.append(image_file)
                processed_images += 1
                
                print(f"  Pose estimated successfully")
            else:
                print(f"  Failed to estimate pose for this image")
        else:
            print(f"  No markers detected in this image")
    
    print(f"Processed {processed_images} images with detected markers")
    
    if len(poses) == 0:
        raise ValueError("No poses could be estimated. Check your images and calibration parameters.")
    
    return poses, images

def visualize_camera_poses(poses, images, scale=0.02):

    print("Starting viser server...")
    server = viser.ViserServer(share=True)
    print(f"Visualizing {len(poses)} camera poses")
    
    # Process each pose
    for i, (c2w, image_file) in enumerate(zip(poses, images)):
        # Read image
        img = cv2.imread(image_file)
        if img is None:
            continue
            
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get image dimensions
        H, W = img.shape[:2]
        
        # Load camera calibration parameters to get intrinsic matrix
        camera_matrix, _ = load_calibration_params()
        K = camera_matrix
        
        # Add camera frustum to the scene
        server.scene.add_camera_frustum(
            f"/cameras/{i}",  # give it a name
            fov=2 * np.arctan2(H / 2, K[0, 0]),  # field of view
            aspect=W / H,  # aspect ratio
            scale=scale,  # scale of the camera frustum
            wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz,  # orientation in quaternion format
            position=c2w[:3, 3],  # position of the camera
            image=img  # image to visualize
        )
    
    print("Visualization started. Press Ctrl+C to exit.")
    print("Take screenshots of the visualization for your deliverables.")
    
    try:
        while True:
            time.sleep(0.1)  # Wait to allow visualization to run
    except KeyboardInterrupt:
        print("Visualization stopped.")

def main():
    
    # Check if object scan images directory exists
    images_path = "object_scan_images"
    if not os.path.exists(images_path):
        print(f"Directory '{images_path}' does not exist.")
        print("Please create this directory and place your object scan images inside.")
        return
    
    # Estimate camera poses
    try:
        print("Starting camera pose estimation...")
        poses, images = estimate_camera_poses(
            images_path=images_path,
            tag_size=0.0535  # 2cm tags - adjust this to match your printed tags
        )
        
        print(f"\nEstimated poses for {len(poses)} images")
        
        # Try to visualize the camera poses
        print("Starting visualization...")
        visualize_camera_poses(poses, images)
        
    except Exception as e:
        print(f"Pose estimation failed: {e}")
        print("Make sure you have enough images with visible ArUco tags and have run calibration.")

if __name__ == "__main__":
    main()