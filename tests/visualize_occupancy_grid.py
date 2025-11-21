"""
Visual Occupancy Grid Viewer
Displays occupancy grid with interactive 2D and 3D views
"""

import numpy as np
import sys
import os
import cv2
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.navigation.occupancy_grid import OccupancyGrid3D

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  Matplotlib not available. Install with: pip install matplotlib")

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("⚠️  Open3D not available. 3D visualization disabled.")


class OccupancyGridVisualizer:
    """Real-time occupancy grid visualizer"""
    
    def __init__(self, grid: OccupancyGrid3D):
        self.grid = grid
        self.running = True
        
        # Create visualization windows
        cv2.namedWindow('Occupancy Grid - Top View', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Occupancy Grid - Side View (XZ)', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Occupancy Grid - Side View (YZ)', cv2.WINDOW_NORMAL)
        
        # Resize windows
        cv2.resizeWindow('Occupancy Grid - Top View', 600, 600)
        cv2.resizeWindow('Occupancy Grid - Side View (XZ)', 600, 300)
        cv2.resizeWindow('Occupancy Grid - Side View (YZ)', 600, 300)
        
        # Color maps
        self.colormap = cv2.COLORMAP_JET
        
    def create_top_view(self) -> np.ndarray:
        """Create top-down view (XY plane)"""
        # Get 2D projection from 0 to 3m height
        grid_2d = self.grid.get_occupancy_2d(z_min=0.0, z_max=3.0)
        
        # Normalize to 0-255
        grid_vis = (grid_2d * 255).astype(np.uint8)
        
        # Apply colormap
        grid_colored = cv2.applyColorMap(grid_vis, self.colormap)
        
        # Rotate to match drone perspective (forward = up)
        grid_colored = cv2.rotate(grid_colored, cv2.ROTATE_90_COUNTERCLOCKWISE)
        grid_colored = cv2.flip(grid_colored, 1)
        
        # Draw center cross (drone position)
        h, w = grid_colored.shape[:2]
        cx, cy = w // 2, h // 2
        cv2.line(grid_colored, (cx - 10, cy), (cx + 10, cy), (0, 255, 0), 2)
        cv2.line(grid_colored, (cx, cy - 10), (cx, cy + 10), (0, 255, 0), 2)
        cv2.circle(grid_colored, (cx, cy), 5, (0, 255, 0), -1)
        
        # Add grid lines
        for i in range(0, w, w // 10):
            cv2.line(grid_colored, (i, 0), (i, h), (128, 128, 128), 1)
        for i in range(0, h, h // 10):
            cv2.line(grid_colored, (0, i), (w, i), (128, 128, 128), 1)
        
        # Add text
        cv2.putText(grid_colored, 'FORWARD', (w//2 - 40, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(grid_colored, f'Resolution: {self.grid.resolution}m', 
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return grid_colored
    
    def create_side_view_xz(self) -> np.ndarray:
        """Create side view (XZ plane - forward and height)"""
        # Get middle slice along Y axis
        slice_xz = self.grid.get_slice(axis='y', index=self.grid.cells_y // 2)
        
        # Normalize
        slice_vis = (slice_xz * 255).astype(np.uint8)
        slice_colored = cv2.applyColorMap(slice_vis, self.colormap)
        
        # Rotate for proper orientation
        slice_colored = cv2.rotate(slice_colored, cv2.ROTATE_90_COUNTERCLOCKWISE)
        slice_colored = cv2.flip(slice_colored, 1)
        
        # Draw ground line
        h, w = slice_colored.shape[:2]
        ground_y = h - 1  # Bottom of image
        cv2.line(slice_colored, (0, ground_y), (w, ground_y), (0, 255, 0), 2)
        
        # Draw drone position
        drone_x = w // 2
        drone_y = ground_y
        cv2.circle(slice_colored, (drone_x, drone_y), 5, (0, 255, 0), -1)
        
        # Add labels
        cv2.putText(slice_colored, 'Side View (X-Z)', (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return slice_colored
    
    def create_side_view_yz(self) -> np.ndarray:
        """Create side view (YZ plane - left/right and height)"""
        # Get middle slice along X axis
        slice_yz = self.grid.get_slice(axis='x', index=self.grid.cells_x // 2)
        
        # Normalize
        slice_vis = (slice_yz * 255).astype(np.uint8)
        slice_colored = cv2.applyColorMap(slice_vis, self.colormap)
        
        # Rotate for proper orientation
        slice_colored = cv2.rotate(slice_colored, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Draw ground line
        h, w = slice_colored.shape[:2]
        ground_y = h - 1
        cv2.line(slice_colored, (0, ground_y), (w, ground_y), (0, 255, 0), 2)
        
        # Draw drone position
        drone_x = w // 2
        drone_y = ground_y
        cv2.circle(slice_colored, (drone_x, drone_y), 5, (0, 255, 0), -1)
        
        # Add labels
        cv2.putText(slice_colored, 'Side View (Y-Z)', (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(slice_colored, 'LEFT', (10, h - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(slice_colored, 'RIGHT', (w - 60, h - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return slice_colored
    
    def update_display(self):
        """Update all visualization windows"""
        # Create views
        top_view = self.create_top_view()
        side_xz = self.create_side_view_xz()
        side_yz = self.create_side_view_yz()
        
        # Display
        cv2.imshow('Occupancy Grid - Top View', top_view)
        cv2.imshow('Occupancy Grid - Side View (XZ)', side_xz)
        cv2.imshow('Occupancy Grid - Side View (YZ)', side_yz)
        
        # Get statistics
        stats = self.grid.get_statistics()
        
        # Print to console
        print(f"\rOccupied: {stats['occupied_cells']:5d} | "
              f"Ratio: {stats['occupancy_ratio']*100:5.2f}% | "
              f"Update: {stats['avg_update_time_ms']:5.2f}ms", end='')
    
    def run(self, point_generator, fps=10):
        """
        Run visualization loop
        
        Args:
            point_generator: Function that returns point cloud array
            fps: Frames per second
        """
        frame_time = 1.0 / fps
        
        print("\n" + "="*70)
        print("OCCUPANCY GRID VISUALIZER")
        print("="*70)
        print("Controls:")
        print("  SPACE - Pause/Resume")
        print("  C     - Clear grid")
        print("  Q/ESC - Quit")
        print("="*70 + "\n")
        
        paused = False
        
        while self.running:
            loop_start = time.time()
            
            if not paused:
                # Get new point cloud
                points = point_generator()
                
                # Update grid
                if points is not None and len(points) > 0:
                    self.grid.update(points)
            
            # Update display
            self.update_display()
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                self.running = False
            elif key == ord(' '):  # SPACE
                paused = not paused
                status = "PAUSED" if paused else "RUNNING"
                print(f"\n{status}")
            elif key == ord('c'):  # C
                self.grid.clear()
                print("\nGrid cleared")
            
            # Maintain frame rate
            elapsed = time.time() - loop_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
        
        cv2.destroyAllWindows()
        print("\n\nVisualization closed.")


def matplotlib_3d_view(grid: OccupancyGrid3D):
    """Create 3D matplotlib visualization (static)"""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available for 3D view")
        return
    
    # Get occupied cells
    occupied = grid.grid > grid.occupancy_threshold
    coords = np.argwhere(occupied)
    
    if len(coords) == 0:
        print("No occupied cells to visualize")
        return
    
    # Convert to world coordinates
    world_coords = coords * grid.resolution
    world_coords[:, 0] -= grid.origin_x
    world_coords[:, 1] -= grid.origin_y
    world_coords[:, 2] -= grid.origin_z
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot occupied cells
    ax.scatter(world_coords[:, 0], world_coords[:, 1], world_coords[:, 2],
               c=grid.grid[coords[:, 0], coords[:, 1], coords[:, 2]],
               cmap='hot', marker='s', s=20, alpha=0.6)
    
    # Plot drone at origin
    ax.scatter([0], [0], [0], c='green', marker='o', s=100, label='Drone')
    
    # Set labels
    ax.set_xlabel('X (Forward) [m]')
    ax.set_ylabel('Y (Left/Right) [m]')
    ax.set_zlabel('Z (Height) [m]')
    ax.set_title('3D Occupancy Grid Visualization')
    ax.legend()
    
    # Set limits
    ax.set_xlim([-grid.size_x/2, grid.size_x/2])
    ax.set_ylim([-grid.size_y/2, grid.size_y/2])
    ax.set_zlim([0, grid.size_z])
    
    plt.show()


# ============ Example Usage ============

def generate_moving_obstacle():
    """Generate moving obstacle for demonstration"""
    t = time.time()
    
    # Oscillating wall
    x_pos = 3.0 + 0.5 * np.sin(t)
    
    points = []
    for _ in range(100):
        x = x_pos + np.random.normal(0, 0.1)
        y = np.random.uniform(-2, 2)
        z = np.random.uniform(0.5, 2.0)
        points.append([x, y, z])
    
    # Add some scattered points
    for _ in range(50):
        x = np.random.uniform(1, 4)
        y = np.random.uniform(-3, 3)
        z = np.random.uniform(0, 2)
        points.append([x, y, z])
    
    return np.array(points)


def load_real_point_cloud():
    """Load real point cloud from file"""
    if not HAS_OPEN3D:
        return None
    
    pcd_path = os.path.join(os.path.dirname(__file__), "kitti-frames_pointcloud.ply")
    
    if not os.path.exists(pcd_path):
        return None
    
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)
    
    # Filter to grid range
    mask = (points[:, 0] >= -5) & (points[:, 0] <= 5) & \
           (points[:, 1] >= -5) & (points[:, 1] <= 5) & \
           (points[:, 2] >= -1) & (points[:, 2] <= 3)
    
    return points[mask]


def demo_synthetic():
    """Demo with synthetic moving obstacles"""
    print("Running demo with synthetic obstacles...")
    
    config = {
        'grid_resolution': 0.1,
        'grid_size_x': 10.0,
        'grid_size_y': 10.0,
        'grid_size_z': 3.0,
        'occupancy_threshold': 0.5,
        'decay_rate': 0.95,  # Slower decay for visualization
        'compute_distance_transform': True
    }
    
    grid = OccupancyGrid3D(config)
    visualizer = OccupancyGridVisualizer(grid)
    
    # Run with moving obstacle
    visualizer.run(generate_moving_obstacle, fps=10)


def demo_real_data():
    """Demo with real KITTI point cloud"""
    print("Running demo with real KITTI data...")
    
    points = load_real_point_cloud()
    
    if points is None:
        print("Could not load real point cloud. Falling back to synthetic demo.")
        demo_synthetic()
        return
    
    print(f"Loaded {len(points)} points from KITTI dataset")
    
    config = {
        'grid_resolution': 0.1,
        'grid_size_x': 10.0,
        'grid_size_y': 10.0,
        'grid_size_z': 3.0,
        'occupancy_threshold': 0.5,
        'decay_rate': 0.98,  # Very slow decay for static scene
        'compute_distance_transform': True
    }
    
    grid = OccupancyGrid3D(config)
    visualizer = OccupancyGridVisualizer(grid)
    
    # Return the same points (static scene)
    visualizer.run(lambda: points, fps=5)


def demo_3d_matplotlib():
    """Demo 3D visualization with matplotlib"""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available")
        return
    
    print("Creating 3D matplotlib visualization...")
    
    config = {
        'grid_resolution': 0.1,
        'grid_size_x': 10.0,
        'grid_size_y': 10.0,
        'grid_size_z': 3.0,
        'occupancy_threshold': 0.5,
        'decay_rate': 0.9,
        'compute_distance_transform': True
    }
    
    grid = OccupancyGrid3D(config)
    
    # Generate some obstacles
    points = []
    
    # Wall
    for _ in range(200):
        x = 3.0 + np.random.normal(0, 0.1)
        y = np.random.uniform(-2, 2)
        z = np.random.uniform(0.5, 2.0)
        points.append([x, y, z])
    
    # Box on left
    for _ in range(100):
        x = 2.0 + np.random.normal(0, 0.1)
        y = -1.5 + np.random.normal(0, 0.1)
        z = 1.0 + np.random.normal(0, 0.2)
        points.append([x, y, z])
    
    points = np.array(points)
    grid.update(points)
    
    matplotlib_3d_view(grid)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Occupancy Grid Visualizer')
    parser.add_argument('--mode', type=str, default='synthetic',
                        choices=['synthetic', 'real', '3d'],
                        help='Visualization mode')
    
    args = parser.parse_args()
    
    if args.mode == 'synthetic':
        demo_synthetic()
    elif args.mode == 'real':
        demo_real_data()
    elif args.mode == '3d':
        demo_3d_matplotlib()
