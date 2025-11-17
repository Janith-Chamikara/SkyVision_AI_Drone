"""
Occupancy Grid Module
Converts 3D point clouds to 3D occupancy grids for obstacle representation
Includes distance transform for fast collision checking
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from typing import Tuple, Optional
import time
from collections import deque


class OccupancyGrid3D:
    """
    3D Occupancy Grid for spatial representation of obstacles.
    
    Discretizes 3D space into voxel cells and marks occupied/free space.
    Optimized for real-time navigation on Jetson.
    """
    
    def __init__(self, config: dict):
        """
        Initialize 3D occupancy grid.
        
        Args:
            config: Configuration containing:
                - grid_resolution: Size of each voxel cell (meters)
                - grid_size_x, grid_size_y, grid_size_z: Grid dimensions (meters)
                - occupancy_threshold: Threshold for marking cell as occupied
                - decay_rate: Rate at which occupancy values decay over time
                - compute_distance_transform: Whether to compute distance transforms
        """
        self.config = config
        
        # Grid parameters
        self.resolution = config.get('grid_resolution', 0.1)  # 10cm cells
        
        # Grid dimensions in meters
        self.size_x = config.get('grid_size_x', 10.0)  # 10m x
        self.size_y = config.get('grid_size_y', 10.0)  # 10m y
        self.size_z = config.get('grid_size_z', 3.0)   # 3m z (height)
        
        # Grid dimensions in cells
        self.cells_x = int(self.size_x / self.resolution)
        self.cells_y = int(self.size_y / self.resolution)
        self.cells_z = int(self.size_z / self.resolution)
        
        # Initialize grid (centered at drone position)
        self.grid = np.zeros((self.cells_x, self.cells_y, self.cells_z), dtype=np.float32)
        
        # Occupancy parameters
        self.occupancy_threshold = config.get('occupancy_threshold', 0.5)
        self.decay_rate = config.get('decay_rate', 0.9)  # Per frame decay
        
        # Distance transform
        self.compute_dt = config.get('compute_distance_transform', True)
        self.distance_transform = None
        
        # Performance tracking
        self.update_times = deque(maxlen=30)
        
        # Origin offset (drone is at center of grid)
        self.origin_x = self.size_x / 2
        self.origin_y = self.size_y / 2
        self.origin_z = 0.0  # Ground level
        
        print(f"[OccupancyGrid3D] Initialized:")
        print(f"  - Resolution: {self.resolution}m per cell")
        print(f"  - Dimensions: {self.cells_x} x {self.cells_y} x {self.cells_z} cells")
        print(f"  - Size: {self.size_x}m x {self.size_y}m x {self.size_z}m")
        print(f"  - Memory: {self.grid.nbytes / 1024:.1f} KB")
    
    def update(self, points: np.ndarray) -> None:
        """
        Update occupancy grid with new point cloud.
        
        Args:
            points: Nx3 array of XYZ coordinates in world frame
        """
        start_time = time.time()
        
        # Apply decay to existing occupancy values
        self.grid *= self.decay_rate
        
        if points is None or len(points) == 0:
            return
        
        # Convert points to grid coordinates
        grid_coords = self._world_to_grid(points)
        
        # Filter valid coordinates
        valid_mask = self._is_valid_coord(grid_coords)
        valid_coords = grid_coords[valid_mask]
        
        if len(valid_coords) == 0:
            return
        
        # Mark cells as occupied
        # Use additive update to accumulate evidence
        for coord in valid_coords:
            x, y, z = coord
            self.grid[x, y, z] = min(1.0, self.grid[x, y, z] + 0.3)
        
        # Compute distance transform if enabled
        if self.compute_dt:
            self._update_distance_transform()
        
        # Track performance
        elapsed = time.time() - start_time
        self.update_times.append(elapsed)
    
    def _world_to_grid(self, points: np.ndarray) -> np.ndarray:
        """
        Convert world coordinates to grid cell indices.
        
        Args:
            points: Nx3 array of XYZ world coordinates
            
        Returns:
            Nx3 array of grid cell indices (i, j, k)
        """
        # Shift origin to grid center
        points_shifted = points.copy()
        points_shifted[:, 0] += self.origin_x
        points_shifted[:, 1] += self.origin_y
        points_shifted[:, 2] += self.origin_z
        
        # Convert to grid indices
        grid_coords = (points_shifted / self.resolution).astype(np.int32)
        
        return grid_coords
    
    def _is_valid_coord(self, coords: np.ndarray) -> np.ndarray:
        """
        Check if grid coordinates are within bounds.
        
        Args:
            coords: Nx3 array of grid indices
            
        Returns:
            N boolean array indicating valid coordinates
        """
        valid_x = (coords[:, 0] >= 0) & (coords[:, 0] < self.cells_x)
        valid_y = (coords[:, 1] >= 0) & (coords[:, 1] < self.cells_y)
        valid_z = (coords[:, 2] >= 0) & (coords[:, 2] < self.cells_z)
        
        return valid_x & valid_y & valid_z
    
    def _update_distance_transform(self) -> None:
        """
        Compute distance transform on occupancy grid.
        
        For each free cell, computes the distance to the nearest occupied cell.
        Enables fast collision checking during planning.
        """
        # Threshold grid to binary occupancy
        binary_grid = (self.grid > self.occupancy_threshold).astype(np.uint8)
        
        # Compute distance transform (distance to nearest obstacle)
        # Invert so free space = 0, occupied = 1
        self.distance_transform = distance_transform_edt(1 - binary_grid) * self.resolution
    
    def get_occupancy_2d(self, z_min: float = 0.0, z_max: float = 2.0) -> np.ndarray:
        """
        Get 2D occupancy grid by projecting 3D grid onto XY plane.
        
        Args:
            z_min: Minimum height to consider (meters)
            z_max: Maximum height to consider (meters)
            
        Returns:
            2D occupancy grid (cells_x x cells_y)
        """
        # Convert heights to grid indices
        z_min_idx = int((z_min + self.origin_z) / self.resolution)
        z_max_idx = int((z_max + self.origin_z) / self.resolution)
        
        # Clamp to valid range
        z_min_idx = max(0, z_min_idx)
        z_max_idx = min(self.cells_z, z_max_idx)
        
        # Take maximum occupancy across height range
        grid_2d = np.max(self.grid[:, :, z_min_idx:z_max_idx], axis=2)
        
        return grid_2d
    
    def is_occupied(self, point: np.ndarray) -> bool:
        """
        Check if a world point is in an occupied cell.
        
        Args:
            point: 3D world coordinate [x, y, z]
            
        Returns:
            True if occupied, False otherwise
        """
        # Convert to grid coordinates
        grid_coord = self._world_to_grid(point.reshape(1, 3))[0]
        
        # Check bounds
        if not self._is_valid_coord(grid_coord.reshape(1, 3))[0]:
            return False
        
        x, y, z = grid_coord
        return self.grid[x, y, z] > self.occupancy_threshold
    
    def get_distance_at_point(self, point: np.ndarray) -> float:
        """
        Get distance to nearest obstacle at a world point.
        
        Args:
            point: 3D world coordinate [x, y, z]
            
        Returns:
            Distance to nearest obstacle in meters (inf if outside grid)
        """
        if self.distance_transform is None:
            return 0.0
        
        # Convert to grid coordinates
        grid_coord = self._world_to_grid(point.reshape(1, 3))[0]
        
        # Check bounds
        if not self._is_valid_coord(grid_coord.reshape(1, 3))[0]:
            return float('inf')
        
        x, y, z = grid_coord
        return self.distance_transform[x, y, z]
    
    def clear(self) -> None:
        """Clear occupancy grid."""
        self.grid.fill(0.0)
        self.distance_transform = None
    
    def get_statistics(self) -> dict:
        """Get grid statistics."""
        occupied_cells = np.sum(self.grid > self.occupancy_threshold)
        total_cells = self.grid.size
        occupancy_ratio = occupied_cells / total_cells
        
        avg_update_time = np.mean(self.update_times) * 1000 if self.update_times else 0.0
        
        return {
            'occupied_cells': int(occupied_cells),
            'total_cells': int(total_cells),
            'occupancy_ratio': float(occupancy_ratio),
            'avg_update_time_ms': float(avg_update_time),
            'grid_shape': self.grid.shape,
            'resolution': self.resolution,
            'size_meters': (self.size_x, self.size_y, self.size_z)
        }
    
    def get_slice(self, axis: str = 'z', index: Optional[int] = None) -> np.ndarray:
        """
        Get a 2D slice of the 3D grid for visualization.
        
        Args:
            axis: 'x', 'y', or 'z'
            index: Cell index along axis (default: middle)
            
        Returns:
            2D slice of occupancy grid
        """
        if index is None:
            if axis == 'x':
                index = self.cells_x // 2
            elif axis == 'y':
                index = self.cells_y // 2
            else:  # z
                index = self.cells_z // 2
        
        if axis == 'x':
            return self.grid[index, :, :]
        elif axis == 'y':
            return self.grid[:, index, :]
        else:  # z
            return self.grid[:, :, index]
