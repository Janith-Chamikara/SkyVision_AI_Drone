import cv2
import numpy as np
import open3d as o3d
from .depth_point_cloud_utils import DepthProcessor

class PointCloudGenerator:
    def __init__(self, weight_path, config, use_cuda=True):
        self.depth_processor = DepthProcessor(weight_path, use_cuda)
        self.pc_config = config['point_cloud']
        
    def process_sequence(self, image_files):
        """Process a sequence of images and generate accumulated point cloud."""
        all_points = []
        all_colors = []

        for idx, image_file in enumerate(image_files):
            # Read frame
            frame = cv2.imread(image_file)
            if frame is None:
                print(f"Failed to read image: {image_file}")
                continue
            
            # Get depth map
            depth_map, resized_frame = self.depth_processor.estimate_depth(frame)
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            
            # Create point cloud for this frame
            points, colors = self._frame_to_points(depth_map, rgb_frame)
            
            if points:
                all_points.extend(points)
                all_colors.extend(colors)
            
            print(f"Processed frame {idx + 1}/{len(image_files)}")

        return np.array(all_points), np.array(all_colors)
    
    def _frame_to_points(self, depth_map, rgb_frame):
        """Convert a single frame's depth map to 3D points."""
        h, w = depth_map.shape
        fx = w / 2  # Approximate focal length
        fy = h / 2
        cx = w / 2  # Principal point
        cy = h / 2
        
        points = []
        colors = []
        
        for v in range(h):
            for u in range(w):
                depth = depth_map[v, u] * self.pc_config['depth_scale']
                if depth > 0:  # Valid depth
                    x = (u - cx) * depth / fx
                    y = (v - cy) * depth / fy
                    z = depth
                    
                    points.append([x, y, z])
                    colors.append(rgb_frame[v, u] / 255.0)
        
        return points, colors

def create_point_cloud(points, colors):
    """Create an Open3D point cloud from points and colors."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def save_point_cloud(pcd, output_path, remove_outliers=True):
    """Save point cloud to file, optionally removing outliers."""
    if remove_outliers:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    o3d.io.write_point_cloud(output_path, pcd)
    return pcd

def visualize_point_cloud(pcd_or_path, window_name="Point Cloud Viewer"):
    """Visualize a point cloud using Open3D."""
    # Load point cloud if path is provided
    if isinstance(pcd_or_path, str):
        pcd = o3d.io.read_point_cloud(pcd_or_path)
        print(f"Loaded point cloud with {len(pcd.points)} points")
    else:
        pcd = pcd_or_path
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name)
    
    # Add the geometry
    vis.add_geometry(pcd)
    
    # Set default camera view
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.set_front([0, 0, -1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, -1, 0])
    
    # Improve visualization
    opt = vis.get_render_option()
    opt.point_size = 1.0
    opt.background_color = [0.1, 0.1, 0.1]  # Dark gray background
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()