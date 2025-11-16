import torch
import torch.nn as nn
import numpy as np


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
            x: Input coordinates of shape (..., 3) or (..., 2)
            
        Returns:
            Encoded coordinates
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


class RaysDataset:
    """
    Dataset for sampling rays from multi-view images
    """
    def __init__(self, images, K, c2ws, device='cuda'):
        """
        Args:
            images: Training images of shape (N, H, W, 3)
            K: Camera intrinsic matrix of shape (3, 3)
            c2ws: Camera-to-world matrices of shape (N, 4, 4)
            device: Device to store the data on
        """
        self.images = torch.tensor(images, dtype=torch.float32, device=device)
        self.K = torch.tensor(K, dtype=torch.float32, device=device)
        self.c2ws = torch.tensor(c2ws, dtype=torch.float32, device=device)
        self.device = device

        self.K = self.K.expand(self.c2ws.shape[0], -1, -1)
        
        self.N, self.H, self.W = images.shape[:3]
        
        # Precompute all rays
        self._precompute_rays()
        
    def _precompute_rays(self):
        """
        Precompute rays for all images
        """
        # Create pixel coordinate grid
        i, j = torch.meshgrid(
            torch.linspace(0, self.W-1, self.W, device=self.device), 
            torch.linspace(0, self.H-1, self.H, device=self.device), 
            indexing='ij'
        )
        i = i.T
        j = j.T
        uv = torch.stack([i, j], dim=-1).reshape(-1, 2)
        uv = uv + 0.5
        uv = uv.unsqueeze(0).expand(self.N, -1, -1)
        
        # Get rays for all pixels
        self.rays_o, self.rays_d = self._get_rays(uv)
        
        # Reshape rays to [N*H*W, 3]
        self.rays_o = self.rays_o.reshape(-1, 3)
        self.rays_d = self.rays_d.reshape(-1, 3)
        # Reshape pixels to [N*H*W, 3]
        self.pixels = self.images.reshape(-1, 3)
        
        # Store uv coordinates for verification
        self.uvs = uv.reshape(-1, 2).to(torch.long)
        
    def _get_rays(self, uv):
        """
        Get rays for given pixel coordinates
        
        Args:
            uv: Pixel coordinates of shape (N, H*W, 2)
            
        Returns:
            rays_o: Ray origins of shape (N, H*W, 3)
            rays_d: Ray directions of shape (N, H*W, 3)
        """
        ray_o = self.c2ws[..., :3, 3]  # Shape: (N, 3)
        ray_o = ray_o.unsqueeze(1).expand(-1, uv.shape[1], -1)  # Shape: (N, H*W, 3)
        
        s = torch.ones_like(uv[..., 0])  # Shape: (N, H*W)
        x_c = self._pixel_to_camera(uv, s)  # Shape: (N, H*W, 3)
        
        x_w = self._transform(x_c)  # Shape: (N, H*W, 3)
        
        ray_d = x_w - ray_o  # Shape: (N, H*W, 3)
        ray_d = torch.nn.functional.normalize(ray_d, dim=-1)
        
        return ray_o, ray_d
    
    def _pixel_to_camera(self, uv, s):
        """
        Transform pixel coordinates to camera coordinates
        
        Args:
            uv: Pixel coordinates of shape (..., 2)
            s: Depth values of shape (...)
            
        Returns:
            x_c: Camera coordinates of shape (..., 3)
        """
        # Extract focal length and principal point from K matrix
        fx = self.K[..., 0, 0]
        fy = self.K[..., 1, 1]
        cx = self.K[..., 0, 2]
        cy = self.K[..., 1, 2]
        
        # Expand to match uv dimensions
        fx = fx.unsqueeze(1).expand(-1, uv.shape[1])  # Shape: (N, H*W)
        fy = fy.unsqueeze(1).expand(-1, uv.shape[1])  # Shape: (N, H*W)
        cx = cx.unsqueeze(1).expand(-1, uv.shape[1])  # Shape: (N, H*W)
        cy = cy.unsqueeze(1).expand(-1, uv.shape[1])  # Shape: (N, H*W)
        
        # Pixel coordinates
        u = uv[..., 0]  # Shape: (N, H*W)
        v = uv[..., 1]  # Shape: (N, H*W)
        
        # Invert the projection equations:
        # u = fx * (x/z) + cx => x = (u - cx) * z / fx
        # v = fy * (y/z) + cy => y = (v - cy) * z / fy
        # z = s
        x = (u - cx) * s / fx
        y = (v - cy) * s / fy
        z = s
        
        # Stack to create 3D points
        x_c = torch.stack([x, y, z], dim=-1)  # Shape: (N, H*W, 3)
        
        return x_c
    
    def _transform(self, x_c):
        """
        Transform points from camera to world space
        
        Args:
            x_c: Points in camera space of shape (N, H*W, 3)
            
        Returns:
            x_w: Points in world space of shape (N, H*W, 3)
        """
        # Convert to homogeneous coordinates
        ones = torch.ones_like(x_c[..., :1])  # Shape: (N, H*W, 1)
        x_c_hom = torch.cat([x_c, ones], dim=-1)  # Shape: (N, H*W, 4)
        
        # Apply transformation: x_w = c2w * x_c
        x_w_hom = torch.matmul(self.c2ws.unsqueeze(1), x_c_hom.unsqueeze(-1)).squeeze(-1)
        
        # Return only the 3D coordinates
        return x_w_hom[..., :3]
    
    def sample_rays(self, N_rays):
        """
        Sample N_rays rays from the dataset
        
        Args:
            N_rays: Number of rays to sample
            
        Returns:
            rays_o: Ray origins of shape (N_rays, 3)
            rays_d: Ray directions of shape (N_rays, 3)
            pixels: Pixel colors of shape (N_rays, 3)
        """
        # Randomly sample indices
        indices = torch.randint(0, self.rays_o.shape[0], (N_rays,), device=self.device)
        
        # Return sampled rays and pixels
        return self.rays_o[indices], self.rays_d[indices], self.pixels[indices]


def sample_along_rays(rays_o, rays_d, near=2.0, far=6.0, N_samples=32, perturb=True):
    """
    Sample points along rays
    
    Args:
        rays_o: Ray origins of shape (..., 3)
        rays_d: Ray directions of shape (..., 3)
        near: Near plane distance
        far: Far plane distance
        N_samples: Number of samples per ray
        perturb: Whether to add random perturbation to samples
        
    Returns:
        points: Sampled points of shape (..., N_samples, 3)
    """
    # Create sample points along rays
    t_vals = torch.linspace(near, far, N_samples, device=rays_o.device)  # Shape: (N_samples,)
    t_vals = t_vals.expand(list(rays_o.shape[:-1]) + [N_samples])  # Shape: (..., N_samples)
    
    # Add random perturbation if requested
    if perturb:
        t_width = (far - near) / N_samples
        t_vals = t_vals + torch.rand_like(t_vals) * t_width
    
    # Compute 3D points
    # points = o + d * t
    rays_o_expanded = rays_o.unsqueeze(-2)  # Shape: (..., 1, 3)
    rays_d_expanded = rays_d.unsqueeze(-2)  # Shape: (..., 1, 3)
    t_vals_expanded = t_vals.unsqueeze(-1)  # Shape: (..., N_samples, 1)
    
    points = rays_o_expanded + rays_d_expanded * t_vals_expanded  # Shape: (..., N_samples, 3)
    
    return points


class NeRF(nn.Module):
    """
    Neural Radiance Field MLP
    """
    def __init__(self, 
                 coord_frequencies=10, 
                 dir_frequencies=4, 
                 hidden_channels=256, 
                 num_layers=8):
        """
        Args:
            coord_frequencies: Number of frequencies for coordinate positional encoding
            dir_frequencies: Number of frequencies for direction positional encoding
            hidden_channels: Width of hidden layers
            num_layers: Number of layers in the MLP
        """
        super(NeRF, self).__init__()
        
        # Positional encodings
        self.coord_encoding = PositionalEncoding(coord_frequencies)
        self.dir_encoding = PositionalEncoding(dir_frequencies)
        
        # Calculate input dimensions after positional encoding
        self.coord_dim = 3 + 2 * 3 * coord_frequencies  # 3 + 2*3*L
        self.dir_dim = 3 + 2 * 3 * dir_frequencies      # 3 + 2*3*L
        
        # Build density MLP (outputs density only)
        self.layers1 = nn.ModuleList()
        self.layers1.append(nn.Linear(self.coord_dim, hidden_channels))
        self.layers1.append(nn.ReLU())
        for i in range(1, num_layers // 2):
            self.layers1.append(nn.Linear(hidden_channels, hidden_channels))
            self.layers1.append(nn.ReLU())
        self.layers1 = nn.Sequential(*self.layers1)

        self.layers2 = nn.ModuleList()
        self.layers2.append(nn.Linear(hidden_channels + self.coord_dim, hidden_channels))
        self.layers2.append(nn.ReLU())
        for i in range(num_layers // 2, num_layers):
            self.layers2.append(nn.Linear(hidden_channels, hidden_channels))
            self.layers2.append(nn.ReLU())
        self.layers2 = nn.Sequential(*self.layers2)
            
        # Output layer for density (with ReLU activation to ensure positive density)
        self.density_output = nn.Sequential(
            nn.Linear(hidden_channels, 1),
            nn.ReLU()
        )
        
        # Feature layer for color prediction
        self.feature_layer = nn.Linear(hidden_channels, hidden_channels)
        
        # Build color MLP (outputs RGB color)
        # Takes intermediate features + encoded direction as input
        self.color_mlp = nn.Sequential(
            nn.Linear(hidden_channels + self.dir_dim, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 3),
            nn.Sigmoid()  # Constrain output to [0, 1]
        )
        
    def forward(self, x, d, intermediate_features=None):
        """
        Forward pass through the NeRF network
        
        Args:
            x: 3D world coordinates of shape (..., 3)
            d: Ray directions of shape (..., 3)
            
        Returns:
            rgb: Predicted colors of shape (..., 3)
            sigma: Predicted densities of shape (..., 1)
        """
        # Apply positional encoding to coordinates
        x_encoded = self.coord_encoding(x)
        
        # Apply positional encoding to directions
        d_encoded = self.dir_encoding(d)
        
        # Pass through density network
        h = self.layers1(x_encoded)
        h = self.layers2(torch.cat([h, x_encoded], dim=-1))
        
        # Extract density
        sigma = self.density_output(h)
        
        # Process features for color prediction
        if intermediate_features is not None:
            h = self.feature_layer(intermediate_features[..., :self.density_layers[0].out_features])
        else:
            h = self.feature_layer(h)
        h = torch.cat([h, d_encoded], dim=-1)
        rgb = self.color_mlp(h)
        
        return rgb, sigma


def volrend(sigmas, rgbs, step_size):
    """
    Volume rendering function
    
    Args:
        sigmas: Densities of shape (B, N_samples, 1)
        rgbs: Colors of shape (B, N_samples, 3)
        step_size: Size of each step along the ray
        
    Returns:
        rendered_colors: Rendered colors of shape (B, 3)
    """
    # Calculate alpha values (probability of absorption)
    # alpha = 1 - exp(-sigma * delta)
    alphas = 1 - torch.exp(-sigmas * step_size)  # Shape: (B, N_samples, 1)
    
    # Shift alphas to the right and set first element to 0
    one_minus_alphas = 1 - alphas  # Shape: (B, N_samples, 1)
    
    # Create tensor for cumulative product (T_i values)
    T = torch.cumprod(
        torch.cat([
            torch.ones_like(one_minus_alphas[:, :1]),  # First element is 1
            one_minus_alphas[:, :-1]  # Shifted alphas
        ], dim=1), dim=1)
    weights = T * alphas  # Shape: (B, N_samples, 1)
    
    # Render colors by weighted sum
    # C = sum(w_i * c_i)
    rendered_colors = torch.sum(weights * rgbs, dim=1)  # Shape: (B, 3)
    
    return rendered_colors


def psnr(mse):
    return -10 * torch.log10(mse + 1e-8)

