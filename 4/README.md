# NeRF Project Documentation

This project implements a Neural Radiance Fields (NeRF) system for 3D scene reconstruction and novel view synthesis. The implementation includes camera calibration, pose estimation, image undistortion, and a complete NeRF training pipeline.

## Project Structure

```
4/
├── part0/              # Camera calibration and data preprocessing
├── part1/              # Neural field implementation
├── part2/              # NeRF implementation and training
├── part2.6/            # Advanced NeRF implementation
├── README.md           # This file
└── ...
```

## Part 0: Camera Calibration and Data Preprocessing

### calibrate_camera.py

**Function**: Calibrates the camera using a chessboard pattern to compute intrinsic camera parameters and distortion coefficients.

**Key Features**:
- Supports chessboard corner detection
- Calculates camera intrinsic matrix and distortion coefficients
- Saves calibration parameters for later use

**How to Run**:
```bash
cd part0
python calibrate_camera.py
```

Ensure you have images with chessboard patterns in the `calibration_images` directory.

### estimate_pose.py

**Function**: Estimates camera poses (extrinsics), computing the transformation matrix from world coordinates to camera coordinates.

**Key Features**:
- Uses ArUco markers to estimate camera poses
- Generates camera-to-world transformation matrices (c2w)
- Works in conjunction with camera calibration parameters

**How to Run**:
```bash
cd part0
python estimate_pose.py
```

Place images containing ArUco markers in the `object_scan_images` directory.

### undistort_and_package.py

**Function**: Undistorts images using camera calibration parameters and packages the data in NeRF training format.

**Key Features**:
- Image undistortion processing
- Generates training/validation/test datasets
- Packages data in NPZ format for NeRF training

**How to Run**:
```bash
cd part0
python undistort_and_package.py
```

## Part 1: Neural Field Implementation

### neural_field.py

**Function**: Implements a basic neural field network for learning implicit representations of 2D images.

**Key Features**:
- Multi-Layer Perceptron (MLP) architecture
- Fourier feature encoding
- Supports image reconstruction tasks

**How to Run**:
```bash
cd part1
python neural_field.py
```

## Part 2: NeRF Implementation

### nerf_components.py

**Function**: Implements core components of the NeRF system, including positional encoding, NeRF network, ray sampling, and volume rendering.

**Key Features**:
- PositionalEncoding: Positional encoding implementation
- NeRF: Core NeRF network model
- sample_along_rays: Sampling points along rays
- volrend: Volume rendering implementation
- RaysDataset: Ray dataset management

**Usage**:
Imported as a module by other scripts.

### complete_nerf_training.py

**Function**: Complete NeRF training pipeline, including training, validation, visualization, and result generation.

**Key Features**:
- Complete training loop
- Validation and metrics calculation
- Training process visualization
- Result rendering and saving

**How to Run**:
```bash
cd part2
python complete_nerf_training.py
```

### generate_results_from_pretrained.py

**Function**: Generates NeRF results using a pre-trained model without retraining.

**Key Features**:
- Loads pre-trained model
- Generates novel view images
- Creates rendering videos

**How to Run**:
```bash
cd part2
python generate_results_from_pretrained.py
```

### viser_visualization.py

**Function**: Uses the Viser library for 3D visualization, showing NeRF scenes and camera positions.

**Key Features**:
- 3D scene visualization
- Camera trajectory display
- Interactive viewer

**How to Run**:
```bash
cd part2
python viser_visualization.py
```

## Part 2.6: Advanced NeRF Implementation

### nerf_components.py

**Function**: NeRF core components in Part 2.6, similar to Part 2 but may include improvements.

**Usage**:
Imported as a module by other scripts.

### complete_nerf_training.py

**Function**: Complete NeRF training pipeline in Part 2.6, may include performance optimizations or feature enhancements.

**How to Run**:
```bash
cd part2.6
python complete_nerf_training.py
```

## Environment Setup

### setup_compatible_env.py

**Function**: Automatically creates a compatible conda environment and installs required dependencies.

**How to Run**:
```bash
python setup_compatible_env.py
```

## Data Format Description

### my_nerf_data.npz

This is the main data format used for NeRF training, containing the following fields:

- `images_train`: Training images [N_train, H, W, 3]
- `c2ws_train`: Training camera poses [N_train, 4, 4]
- `images_val`: Validation images [N_val, H, W, 3]
- `c2ws_val`: Validation camera poses [N_val, 4, 4]
- `c2ws_test`: Test camera poses [N_test, 4, 4]
- `focal`: Camera focal length (float)

## Recommended Execution Order

1. Camera calibration: `part0/calibrate_camera.py`
2. Pose estimation: `part0/estimate_pose.py`
3. Data processing: `part0/undistort_and_package.py`
4. NeRF training: `part2/complete_nerf_training.py` or `part2.6/complete_nerf_training.py`
5. Result visualization: `part2/viser_visualization.py`

## Important Notes

1. Ensure all required libraries are properly installed
2. Image paths and filenames need to match those specified in the code
3. Training NeRF may take a long time and require significant computational resources
4. If you encounter OpenMP library conflicts, set the environment variable `KMP_DUPLICATE_LIB_OK=TRUE`

## Handling Large Files

This project generates large data files (especially `my_nerf_data.npz` which can exceed 900MB) that cannot be committed to GitHub due to size limitations. These files are automatically excluded from git via the `.gitignore` file.

To share large files with collaborators:

1. Use [Git LFS](https://git-lfs.github.com/) for versioning large files:
   ```bash
   git lfs install
   git lfs track "*.npz"
   git add .gitattributes
   ```

2. Alternatively, share files through cloud storage services like Google Drive, Dropbox, or OneDrive.

3. For model files, consider using model sharing platforms like Hugging Face Model Hub.

The following file types are excluded from git:
- Data files: `*.npz`, `*.npy`, `*.pth`, `*.pt`, `*.pkl`, `*.h5`, `*.hdf5`
- Compressed files: `*.zip`, `*.7z`, `*.tar.gz`
- Rendered outputs: `rendered_images/`, `spherical_rendering/`
- Model checkpoints: `*.ckpt`, `checkpoints/`