"""
Visualization module for Gaussian-to-BEV pipeline

This module provides visualization functions for:
1. fused_agents_gaussians (before pooling)
2. pooled_gaussians (after pooling)
3. voxel_coords (voxel coordinates)
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Optional, Tuple
from datetime import datetime


def visualize_gaussians(
    gaussians: Dict[str, torch.Tensor],
    save_dir: str,
    prefix: str = "gaussians",
    point_cloud_range: Optional[Tuple[float, ...]] = None,
    max_points: int = 10000,
    show_ellipsoids: bool = False
) -> str:
    """
    Visualize Gaussian points in 3D space.
    
    Args:
        gaussians: Dictionary containing:
            - 'mu': [N, 3] - Gaussian centers (x, y, z)
            - 'scale': [N, 3] - Gaussian scales
            - 'rotation': [N, 4] - Gaussian rotations (quaternion)
            - 'features': [N, C] - Gaussian features (optional, for coloring)
            - 'semantic': [N, 2] - Semantic features (optional, for coloring)
        save_dir: Directory to save visualization
        prefix: Prefix for output filename
        point_cloud_range: (x_min, y_min, z_min, x_max, y_max, z_max) for setting axis limits
        max_points: Maximum number of points to visualize (for performance)
        show_ellipsoids: Whether to draw ellipsoids (slower but more informative)
    
    Returns:
        Path to saved visualization file
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to numpy
    if isinstance(gaussians['mu'], torch.Tensor):
        mu = gaussians['mu'].detach().cpu().numpy()
    else:
        mu = np.array(gaussians['mu'])
    
    if len(mu) == 0:
        print(f"Warning: No gaussians to visualize for {prefix}")
        return ""
    
    # Limit number of points for performance
    if len(mu) > max_points:
        indices = np.random.choice(len(mu), max_points, replace=False)
        mu = mu[indices]
        print(f"Warning: Downsampled to {max_points} points for visualization")
    
    # Extract coordinates
    x, y, z = mu[:, 0], mu[:, 1], mu[:, 2]
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color by feature magnitude if available
    if 'features' in gaussians and len(gaussians['features']) > 0:
        if isinstance(gaussians['features'], torch.Tensor):
            features = gaussians['features'].detach().cpu().numpy()
        else:
            features = np.array(gaussians['features'])
        if len(features) > len(mu):
            features = features[:len(mu)]
        # Use feature magnitude for coloring
        feature_mag = np.linalg.norm(features, axis=1)
        scatter = ax.scatter(x, y, z, c=feature_mag, cmap='viridis', s=10, alpha=0.6)
        plt.colorbar(scatter, ax=ax, label='Feature Magnitude')
    elif 'semantic' in gaussians and len(gaussians['semantic']) > 0:
        if isinstance(gaussians['semantic'], torch.Tensor):
            semantic = gaussians['semantic'].detach().cpu().numpy()
        else:
            semantic = np.array(gaussians['semantic'])
        if len(semantic) > len(mu):
            semantic = semantic[:len(mu)]
        # Use first semantic channel for coloring
        scatter = ax.scatter(x, y, z, c=semantic[:, 0], cmap='coolwarm', s=10, alpha=0.6)
        plt.colorbar(scatter, ax=ax, label='Semantic Channel 0')
    else:
        # Default: color by height (z)
        scatter = ax.scatter(x, y, z, c=z, cmap='coolwarm', s=10, alpha=0.6)
        plt.colorbar(scatter, ax=ax, label='Height (Z)')
    
    # Draw ellipsoids if requested (only for a subset to avoid clutter)
    if show_ellipsoids and 'scale' in gaussians:
        if isinstance(gaussians['scale'], torch.Tensor):
            scale = gaussians['scale'].detach().cpu().numpy()
        else:
            scale = np.array(gaussians['scale'])
        if len(scale) > len(mu):
            scale = scale[:len(mu)]
        
        # Only draw ellipsoids for a small subset
        n_ellipsoids = min(50, len(mu))
        ellipsoid_indices = np.random.choice(len(mu), n_ellipsoids, replace=False)
        for idx in ellipsoid_indices:
            _draw_ellipsoid(ax, mu[idx], scale[idx], alpha=0.2, color='red')
    
    # Set axis limits
    if point_cloud_range is not None:
        x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
    else:
        # Auto-scale based on data
        margin = 0.1
        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        z_range = z.max() - z.min()
        ax.set_xlim(x.min() - margin * x_range, x.max() + margin * x_range)
        ax.set_ylim(y.min() - margin * y_range, y.max() + margin * y_range)
        ax.set_zlim(z.min() - margin * z_range, z.max() + margin * z_range)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title(f'{prefix.capitalize()} - {len(mu)} Gaussians', fontsize=14)
    
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Visualization saved to: {save_path}")
    return save_path


def visualize_voxel_coords(
    voxel_coords: torch.Tensor,
    save_dir: str,
    prefix: str = "voxel_coords",
    point_cloud_range: Optional[Tuple[float, ...]] = None,
    voxel_size: Optional[Tuple[float, ...]] = None,
    max_voxels: int = 50000
) -> str:
    """
    Visualize voxel coordinates in 3D space.
    
    Args:
        voxel_coords: [M, 4] tensor of (batch_idx, z, y, x) voxel indices
        save_dir: Directory to save visualization
        prefix: Prefix for output filename
        point_cloud_range: (x_min, y_min, z_min, x_max, y_max, z_max) for converting indices to world coords
        voxel_size: (vx, vy, vz) voxel size for converting indices to world coords
        max_voxels: Maximum number of voxels to visualize
    
    Returns:
        Path to saved visualization file
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to numpy
    if isinstance(voxel_coords, torch.Tensor):
        voxel_coords_np = voxel_coords.detach().cpu().numpy()
    else:
        voxel_coords_np = np.array(voxel_coords)
    
    if len(voxel_coords_np) == 0:
        print(f"Warning: No voxel coordinates to visualize for {prefix}")
        return ""
    
    # Limit number of voxels for performance
    if len(voxel_coords_np) > max_voxels:
        indices = np.random.choice(len(voxel_coords_np), max_voxels, replace=False)
        voxel_coords_np = voxel_coords_np[indices]
        print(f"Warning: Downsampled to {max_voxels} voxels for visualization")
    
    # Extract indices: (batch_idx, z, y, x)
    b, z_idx, y_idx, x_idx = voxel_coords_np[:, 0], voxel_coords_np[:, 1], voxel_coords_np[:, 2], voxel_coords_np[:, 3]
    
    # Convert to world coordinates if range and size are provided
    if point_cloud_range is not None and voxel_size is not None:
        x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
        vx, vy, vz = voxel_size
        # Voxel center = min + (idx + 0.5) * voxel_size
        x = x_min + (x_idx.astype(float) + 0.5) * vx
        y = y_min + (y_idx.astype(float) + 0.5) * vy
        z = z_min + (z_idx.astype(float) + 0.5) * vz
        coord_label = "World Coordinates (m)"
    else:
        # Use indices directly
        x, y, z = x_idx, y_idx, z_idx
        coord_label = "Voxel Indices"
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color by batch index if multiple batches
    unique_batches = np.unique(b)
    if len(unique_batches) > 1:
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_batches)))
        batch_color_map = {batch: colors[i] for i, batch in enumerate(unique_batches)}
        point_colors = [batch_color_map[batch] for batch in b]
        scatter = ax.scatter(x, y, z, c=point_colors, s=20, alpha=0.6)
    else:
        # Color by height (z)
        scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=20, alpha=0.6)
        plt.colorbar(scatter, ax=ax, label='Height')
    
    # Set axis limits
    if point_cloud_range is not None:
        x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
    else:
        # Auto-scale
        margin = 0.1
        x_range = x.max() - x.min() if x.max() > x.min() else 1
        y_range = y.max() - y.min() if y.max() > y.min() else 1
        z_range = z.max() - z.min() if z.max() > z.min() else 1
        ax.set_xlim(x.min() - margin * x_range, x.max() + margin * x_range)
        ax.set_ylim(y.min() - margin * y_range, y.max() + margin * y_range)
        ax.set_zlim(z.min() - margin * z_range, z.max() + margin * z_range)
    
    ax.set_xlabel(f'X ({coord_label})', fontsize=12)
    ax.set_ylabel(f'Y ({coord_label})', fontsize=12)
    ax.set_zlabel(f'Z ({coord_label})', fontsize=12)
    ax.set_title(f'{prefix.capitalize()} - {len(voxel_coords_np)} Voxels', fontsize=14)
    
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Visualization saved to: {save_path}")
    return save_path


def _draw_ellipsoid(ax, center, radii, alpha=0.25, color='red', n_steps=16):
    """
    Draw an ellipsoid in 3D space.
    
    Args:
        ax: 3D axes object
        center: [3] center coordinates
        radii: [3] radii along x, y, z axes
        alpha: Transparency
        color: Color
        n_steps: Number of steps for mesh generation
    """
    u = np.linspace(0, 2 * np.pi, n_steps)
    v = np.linspace(0, np.pi, n_steps)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    
    # Translate to center
    x += center[0]
    y += center[1]
    z += center[2]
    
    ax.plot_surface(x, y, z, alpha=alpha, color=color, linewidth=0, antialiased=False)


def visualize_gaussian2bev_pipeline(
    fused_gaussians: Dict[str, torch.Tensor],
    pooled_gaussians: Dict[str, torch.Tensor],
    voxel_coords: torch.Tensor,
    save_dir: str,
    point_cloud_range: Optional[Tuple[float, ...]] = None,
    voxel_size: Optional[Tuple[float, ...]] = None,
    max_points: int = 10000,
    max_voxels: int = 50000
) -> Dict[str, str]:
    """
    Visualize all three stages of the Gaussian-to-BEV pipeline.
    
    Args:
        fused_gaussians: Dictionary of fused gaussians (before pooling)
        pooled_gaussians: Dictionary of pooled gaussians (after pooling)
        voxel_coords: [M, 4] voxel coordinates
        save_dir: Directory to save visualizations
        point_cloud_range: (x_min, y_min, z_min, x_max, y_max, z_max)
        voxel_size: (vx, vy, vz) voxel size
        max_points: Maximum points to visualize for gaussians
        max_voxels: Maximum voxels to visualize
    
    Returns:
        Dictionary mapping visualization type to saved file path
    """
    os.makedirs(save_dir, exist_ok=True)
    
    saved_paths = {}
    
    # 1. Visualize fused gaussians (before pooling)
    if fused_gaussians and len(fused_gaussians.get('mu', [])) > 0:
        path = visualize_gaussians(
            fused_gaussians,
            save_dir=save_dir,
            prefix="fused_agents_gaussians",
            point_cloud_range=point_cloud_range,
            max_points=max_points,
            show_ellipsoids=False
        )
        saved_paths['fused_gaussians'] = path
    
    # 2. Visualize pooled gaussians (after pooling)
    if pooled_gaussians and len(pooled_gaussians.get('mu', [])) > 0:
        path = visualize_gaussians(
            pooled_gaussians,
            save_dir=save_dir,
            prefix="pooled_gaussians",
            point_cloud_range=point_cloud_range,
            max_points=max_points,
            show_ellipsoids=False
        )
        saved_paths['pooled_gaussians'] = path
    
    # 3. Visualize voxel coordinates
    if voxel_coords is not None and len(voxel_coords) > 0:
        path = visualize_voxel_coords(
            voxel_coords,
            save_dir=save_dir,
            prefix="voxel_coords",
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=max_voxels
        )
        saved_paths['voxel_coords'] = path
    
    return saved_paths



