import cv2
import numpy as np
import glob
import os

def calibrate_camera(images_path, tag_size=0.076):
    """
    Calibrate camera using ArUco tags in images
    
    Parameters:
    images_path: path to folder containing calibration images
    tag_size: size of the ArUco tag in meters (default 0.02m = 2cm)
    
    Returns:
    camera_matrix: intrinsic camera matrix
    dist_coeffs: distortion coefficients
    rvecs: rotation vectors for each image
    tvecs: translation vectors for each image
    """
    
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
    
    # Lists to store object points and image points from all images
    all_object_points = []  # 3D points in real world space
    all_image_points = []   # 2D points in image plane
    
    # Get list of image files
    image_files = glob.glob(os.path.join(images_path, "*.jpg")) + \
                  glob.glob(os.path.join(images_path, "*.png")) + \
                  glob.glob(os.path.join(images_path, "*.jpeg"))
    
    print(f"Found {len(image_files)} images for calibration")
    
    # Process each image
    processed_images = 0
    for idx, image_file in enumerate(image_files):
        print(f"Processing image {idx+1}/{len(image_files)}: {os.path.basename(image_file)}")
        
        image = cv2.imread(image_file)
        if image is None:
            print(f"Warning: Could not read image {image_file}")
            continue
            
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers in the image
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
        
        # Check if any markers were detected
        if ids is not None:
            print(f"  Detected {len(ids)} markers")
            
            # Process each detected marker
            for i in range(len(ids)):
                # Add object points (same for each tag since they represent the same 3D points)
                all_object_points.append(object_points)
                
                # Add image points (corners of the detected tag)
                image_points = corners[i][0].astype(np.float32)
                all_image_points.append(image_points)
                
            processed_images += 1
        else:
            print(f"  No markers detected in this image")
    
    print(f"Processed {processed_images} images with detected markers")
    print(f"Total points for calibration: {len(all_object_points)}")
    
    # Check if we have any detections
    if len(all_object_points) == 0:
        raise ValueError("No ArUco markers detected in any images. Calibration failed.")
    
    # Get image dimensions from the last processed image
    height, width = gray.shape
    
    # Calibrate camera
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        all_object_points, 
        all_image_points, 
        (width, height), 
        None, 
        None
    )
    
    print("\nCalibration completed!")
    print(f"Reprojection error: {rms}")
    print("Camera matrix:")
    print(camera_matrix)
    print("Distortion coefficients:")
    print(dist_coeffs)
    
    return camera_matrix, dist_coeffs, rvecs, tvecs

def save_calibration_params(camera_matrix, dist_coeffs, filename="camera_params.npz"):
    np.savez(filename, 
             camera_matrix=camera_matrix, 
             dist_coeffs=dist_coeffs)
    print(f"Calibration parameters saved to {filename}")

def load_calibration_params(filename="camera_params.npz"):
    data = np.load(filename)
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']
    print(f"Calibration parameters loaded from {filename}")
    return camera_matrix, dist_coeffs

def main():
    
    images_path = "calibration_images"
    if not os.path.exists(images_path):
        print(f"Directory '{images_path}' does not exist.")
        return

    try:
        print("Starting camera calibration...")
        camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera(
            images_path=images_path,
            tag_size=0.076
        )
        
        save_calibration_params(camera_matrix, dist_coeffs, 'camera_params.npz')
        
        print("\nCalibration successful!")
        
    except Exception as e:
        print(f"Calibration failed: {e}")

if __name__ == "__main__":
    main()