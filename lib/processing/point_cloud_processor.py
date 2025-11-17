"""
Point Cloud Processing Module
Implements voxel downsampling, statistical outlier removal, and temporal filtering
Optimized for Jetson GPU acceleration
"""

import numpy as np
import open3d as o3d
from collections import deque
from typing import Tuple, Optional
import time


class PointCloudProcessor:
    """
    Real-time point cloud processing with Jetson optimization.
    
    Features:
    - Voxel grid downsampling
    - Statistical outlier removal
    - Temporal smoothing
    - Range filtering
    """
    
    def __init__(self, config: dict):
        """
        Initialize point cloud processor.
        
        Args:
            config: Configuration dictionary containing:
                - voxel_size: Size of voxel for downsampling (meters)
                - max_detection_range: Maximum distance for points (meters)
                - nb_neighbors: Number of neighbors for outlier removal
                - std_ratio: Standard deviation threshold for outliers
                - temporal_buffer_size: Number of frames for temporal smoothing
                - min_points: Minimum number of points to process
        """
        self.config = config
        
        # Voxel downsampling
        self.voxel_size = config.get('voxel_size', 0.05)  # 5cm voxels
        
        # Range filtering
        self.max_range = config.get('max_detection_range', 10.0)
        self.min_range = config.get('min_detection_range', 0.3)
        
        # Statistical outlier removal
        self.nb_neighbors = config.get('nb_neighbors', 20)
        self.std_ratio = config.get('std_ratio', 2.0)
        
        # Temporal smoothing
        self.temporal_buffer_size = config.get('temporal_buffer_size', 3)
        self.point_buffer = deque(maxlen=self.temporal_buffer_size)
        
        # Minimum points threshold
        self.min_points = config.get('min_points', 100)
        
        # Performance tracking
        self.processing_times = deque(maxlen=30)
        
        # CUDA availability (for future optimization)
        self.use_cuda = config.get('use_cuda', False)
        
        print(f"[PointCloudProcessor] Initialized with:")
        print(f"  - Voxel size: {self.voxel_size}m")
        print(f"  - Range: {self.min_range}-{self.max_range}m")
        print(f"  - Outlier removal: {self.nb_neighbors} neighbors, {self.std_ratio} std")
        print(f"  - Temporal buffer: {self.temporal_buffer_size} frames")
    
    def process(self, points: np.ndarray, colors: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete point cloud processing pipeline.
        
        Args:
            points: Nx3 array of XYZ coordinates
            colors: Nx3 array of RGB colors (optional)
            
        Returns:
            Tuple of (filtered_points, filtered_colors)
        """
        start_time = time.time()
        
        if points is None or len(points) < self.min_points:
            return np.array([]), np.array([])
        
        # Step 1: Range filtering (fast, do first)
        points, colors = self._filter_by_range(points, colors)
        
        if len(points) < self.min_points:
            return np.array([]), np.array([])
        
        # Step 2: Voxel downsampling (reduces computation for later steps)
        points, colors = self._voxel_downsample(points, colors)
        
        if len(points) < self.min_points:
            return np.array([]), np.array([])
        
        # Step 3: Statistical outlier removal
        points, colors = self._remove_outliers(points, colors)
        
        if len(points) < self.min_points:
            return np.array([]), np.array([])
        
        # Step 4: Temporal smoothing
        points, colors = self._temporal_smooth(points, colors)
        
        # Track performance
        elapsed = time.time() - start_time
        self.processing_times.append(elapsed)
        
        return points, colors
    
    def _filter_by_range(self, points: np.ndarray, colors: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter points by distance range from origin.
        Removes points too close (noise) or too far (unreliable).
        """
        if len(points) == 0:
            return points, colors if colors is not None else np.array([])
        
        # Compute distances from origin (drone position)
        distances = np.linalg.norm(points, axis=1)
        
        # Create mask for valid range
        valid_mask = (distances >= self.min_range) & (distances <= self.max_range)
        
        # Filter points and colors
        filtered_points = points[valid_mask]
        filtered_colors = colors[valid_mask] if colors is not None else None
        
        return filtered_points, filtered_colors
    
    def _voxel_downsample(self, points: np.ndarray, colors: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Voxel grid downsampling to reduce point count while preserving geometry.
        
        Divides space into 3D voxels and keeps one representative point per voxel.
        Typical reduction: 500k -> 50-100k points with 0.05m voxels.
        """
        if len(points) == 0:
            return points, colors if colors is not None else np.array([])
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if colors is not None and len(colors) > 0:
            # Normalize colors to [0, 1] if needed
            if colors.max() > 1.0:
                colors_normalized = colors / 255.0
            else:
                colors_normalized = colors
            pcd.colors = o3d.utility.Vector3dVector(colors_normalized)
        
        # Voxel downsampling
        pcd_down = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        
        # Extract downsampled points and colors
        points_down = np.asarray(pcd_down.points)
        colors_down = np.asarray(pcd_down.colors) if pcd_down.has_colors() else None
        
        # Convert colors back to [0, 255] range if needed
        if colors_down is not None and colors is not None and colors.max() > 1.0:
            colors_down = (colors_down * 255).astype(np.uint8)
        
        return points_down, colors_down
    
    def _remove_outliers(self, points: np.ndarray, colors: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Statistical outlier removal using nearest neighbor analysis.
        
        For each point, compute mean distance to k nearest neighbors.
        Remove points whose mean distance is > threshold (mean + std_ratio * std).
        """
        if len(points) < self.nb_neighbors:
            return points, colors if colors is not None else np.array([])
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if colors is not None and len(colors) > 0:
            # Normalize colors to [0, 1] if needed
            if colors.max() > 1.0:
                colors_normalized = colors / 255.0
            else:
                colors_normalized = colors
            pcd.colors = o3d.utility.Vector3dVector(colors_normalized)
        
        # Statistical outlier removal
        pcd_clean, inlier_indices = pcd.remove_statistical_outlier(
            nb_neighbors=self.nb_neighbors,
            std_ratio=self.std_ratio
        )
        
        # Extract cleaned points and colors
        points_clean = np.asarray(pcd_clean.points)
        colors_clean = np.asarray(pcd_clean.colors) if pcd_clean.has_colors() else None
        
        # Convert colors back to [0, 255] range if needed
        if colors_clean is not None and colors is not None and colors.max() > 1.0:
            colors_clean = (colors_clean * 255).astype(np.uint8)
        
        return points_clean, colors_clean
    
    def _temporal_smooth(self, points: np.ndarray, colors: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Temporal smoothing by averaging points across consecutive frames.
        
        Maintains a buffer of recent point clouds and combines them to reduce
        temporal noise and improve obstacle detection stability.
        """
        if len(points) == 0:
            return points, colors if colors is not None else np.array([])
        
        # Add current frame to buffer
        self.point_buffer.append((points, colors))
        
        # If buffer not full yet, return current frame
        if len(self.point_buffer) < self.temporal_buffer_size:
            return points, colors
        
        # Combine all frames in buffer
        all_points = []
        all_colors = []
        
        for frame_points, frame_colors in self.point_buffer:
            all_points.append(frame_points)
            if frame_colors is not None:
                all_colors.append(frame_colors)
        
        # Concatenate
        combined_points = np.vstack(all_points)
        combined_colors = np.vstack(all_colors) if all_colors else None
        
        # Apply voxel downsampling to combined cloud to keep size manageable
        # Use slightly larger voxel size for temporal smoothing
        temporal_voxel_size = self.voxel_size * 1.5
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(combined_points)
        
        if combined_colors is not None:
            if combined_colors.max() > 1.0:
                colors_normalized = combined_colors / 255.0
            else:
                colors_normalized = combined_colors
            pcd.colors = o3d.utility.Vector3dVector(colors_normalized)
        
        pcd_smoothed = pcd.voxel_down_sample(voxel_size=temporal_voxel_size)
        
        smoothed_points = np.asarray(pcd_smoothed.points)
        smoothed_colors = np.asarray(pcd_smoothed.colors) if pcd_smoothed.has_colors() else None
        
        if smoothed_colors is not None and combined_colors is not None and combined_colors.max() > 1.0:
            smoothed_colors = (smoothed_colors * 255).astype(np.uint8)
        
        return smoothed_points, smoothed_colors
    
    def reset_temporal_buffer(self):
        """Clear temporal smoothing buffer."""
        self.point_buffer.clear()
    
    def get_average_processing_time(self) -> float:
        """Get average processing time in milliseconds."""
        if not self.processing_times:
            return 0.0
        return np.mean(self.processing_times) * 1000  # Convert to ms
    
    def get_statistics(self) -> dict:
        """Get processing statistics."""
        return {
            'avg_processing_time_ms': self.get_average_processing_time(),
            'voxel_size': self.voxel_size,
            'detection_range': f"{self.min_range}-{self.max_range}m",
            'temporal_buffer_size': len(self.point_buffer),
            'outlier_removal_params': {
                'nb_neighbors': self.nb_neighbors,
                'std_ratio': self.std_ratio
            }
        }
