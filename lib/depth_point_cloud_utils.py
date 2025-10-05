import numpy as np
import open3d as o3d
import cv2
import torch
from networks.RTMonoDepth.RTMonoDepth import DepthDecoder, DepthEncoder

class DepthProcessor:
    def __init__(self, weight_path, use_cuda=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        
        # Load models
        self.encoder = DepthEncoder()
        self.depth_decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc)
        
        # Load encoder
        encoder_dict = torch.load(f"{weight_path}/encoder.pth", map_location=self.device)
        self.encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in self.encoder.state_dict()})
        self.encoder.to(self.device).eval()
        
        # Load decoder
        self.depth_decoder.load_state_dict(torch.load(f"{weight_path}/depth.pth", map_location=self.device))
        self.depth_decoder.to(self.device).eval()

    def estimate_depth(self, frame):
        """Process a single frame to generate depth map."""
        with torch.no_grad():
            # Prepare input
            input_image = cv2.resize(frame, (640, 192))
            input_tensor = torch.from_numpy(input_image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            input_tensor = input_tensor.to(self.device)

            # Generate depth
            features = self.encoder(input_tensor)
            outputs = self.depth_decoder(features)
            depth = outputs[("disp", 0)]

            # Convert to numpy
            depth_map = depth.squeeze().cpu().numpy()
            return depth_map, input_image

class PointCloudVisualizer:
    def __init__(self, config):
        """Initialize visualizer with configuration parameters."""
        self.config = config['point_cloud']
        self.vis_config = config['visualization']
        
        # Initialize visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            window_name='RT-MonoDepth Point Cloud',
            width=self.config['window_width'],
            height=self.config['window_height']
        )
        self.point_cloud = o3d.geometry.PointCloud()
        
        # Set visualization parameters
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray(self.config['background_color'])
        opt.point_size = self.config['point_size']
        opt.show_coordinate_frame = self.vis_config['show_coordinate_frame']
        
        # Initialize with coordinate frame points
        initial_points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        initial_colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
        self.point_cloud.points = o3d.utility.Vector3dVector(initial_points)
        self.point_cloud.colors = o3d.utility.Vector3dVector(initial_colors)
        
        self.vis.add_geometry(self.point_cloud)
        
        # Set initial view
        self.reset_view()

    def reset_view(self):
        view_control = self.vis.get_view_control()
        view_control.set_zoom(0.8)
        view_control.set_front([0, 0, -1])
        view_control.set_lookat([0, 0, 0])
        view_control.set_up([0, -1, 0])

    def create_point_cloud(self, rgb_image, depth_map):
        """Create point cloud from RGB image and depth map."""
        height, width = depth_map.shape
        fx = width * self.config['focal_scale']
        fy = width * self.config['focal_scale']
        cx = width / 2
        cy = height / 2

        # Create meshgrid of coordinates
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        # Calculate 3D coordinates
        Z = depth_map.reshape(-1)
        X = (x.reshape(-1) - cx) * Z / fx
        Y = (y.reshape(-1) - cy) * Z / fy

        # Stack coordinates
        points = np.stack([X, Y, Z], axis=1)
        colors = rgb_image.reshape(-1, 3) / 255.0
        
        # Filter points based on depth range
        mask = (Z > self.config['min_depth']) & (Z < self.config['max_depth'])
        points = points[mask]
        colors = colors[mask]

        # Scale points for better visualization
        points = points * self.config['depth_scale']
        
        # Center the point cloud
        points[:, 2] -= np.mean(points[:, 2])

        # Subsample points for better performance
        subsample = self.config['subsample_factor']
        if subsample > 1:
            points = points[::subsample]
            colors = colors[::subsample]

        return points, colors

    def update(self, points, colors):
        """Update point cloud visualization."""
        if len(points) > 0:
            self.point_cloud.points = o3d.utility.Vector3dVector(points)
            self.point_cloud.colors = o3d.utility.Vector3dVector(colors)
            self.vis.update_geometry(self.point_cloud)
            self.vis.poll_events()
            self.vis.update_renderer()

    def close(self):
        """Clean up resources."""
        self.vis.destroy_window()