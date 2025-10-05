import cv2
import json
import os
import glob
import numpy as np
from lib.depth_point_cloud_utils import DepthProcessor
import argparse
import open3d as o3d

class SequenceProcessor:
    def __init__(self, weight_path, config_path="config.json", use_cuda=True):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        self.depth_processor = DepthProcessor(weight_path, use_cuda)

        self.video_config = self.config['video']
        self.model_config = self.config['model']
        self.pc_config = self.config['point_cloud']
        
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.video_dir = os.path.join(self.base_dir, 'depth_videos')
        self.pointcloud_dir = os.path.join(self.base_dir, 'point_clouds')
        os.makedirs(self.video_dir, exist_ok=True)
        os.makedirs(self.pointcloud_dir, exist_ok=True)
        
    def process_sequence(self, input_path, image_format='jpg'):
        """Process a sequence of images to create point cloud and optionally save depth video."""
        
        input_dir_name = os.path.basename(os.path.normpath(input_path))
        video_path = os.path.join(self.video_dir, f'{input_dir_name}_depth.mp4')
        pointcloud_path = os.path.join(self.pointcloud_dir, f'{input_dir_name}_pointcloud.ply')
        image_files = sorted(glob.glob(os.path.join(input_path, f'*.{image_format}')))
        
        if not image_files:
            print(f"No {image_format} files found in {input_path}")
            return
            
        print(f"Found {len(image_files)} images to process")
        
        all_points = []
        all_colors = []
        
        video_writer = None
        if video_path:
            first_frame = cv2.imread(image_files[0])
            height, width = first_frame.shape[:2]
            
            video_dir = os.path.dirname(video_path)
            if video_dir:
                os.makedirs(video_dir, exist_ok=True)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                video_path,
                fourcc,
                30.0, 
                (width * 2, height) 
            )

        try:
            window_name = 'RGB and Depth Visualization'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1280, 480) 
            
            print("\nProcessing frames...")
            print("Press 'Q' to stop early")
            
            for idx, image_file in enumerate(image_files):
                frame = cv2.imread(image_file)
                if frame is None:
                    print(f"Failed to read image: {image_file}")
                    continue
                
                depth_map, resized_frame = self.depth_processor.estimate_depth(frame)
                
                depth_colormap = cv2.applyColorMap(
                    cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U),
                    cv2.COLORMAP_MAGMA
                )
                
                depth_display = cv2.resize(depth_colormap, (frame.shape[1], frame.shape[0]))

                if self.pc_config['enabled']:
                    h, w = depth_map.shape
                    fx =  self.video_config['focal_length_x']
                    fy =  self.video_config['focal_length_y']
                    cx =  self.video_config['c_x']
                    cy =  self.video_config['c_y']
                    
                    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                    frame_points = []
                    frame_colors = []
                    
                    for v in range(h):
                        for u in range(w):
                            depth = depth_map[v, u] * self.pc_config['depth_scale']
                            if depth > 0: 
                                x = (u - cx) * depth / fx
                                y = (v - cy) * depth / fy
                                z = depth
                                
                                frame_points.append([x, y, z])
                                frame_colors.append(rgb_frame[v, u] / 255.0)
                    
                    if frame_points:
                        all_points.extend(frame_points)
                        all_colors.extend(frame_colors)
                        print(f"Frame {idx + 1}/{len(image_files)} - Points: {len(frame_points)}")

                combined_display = np.hstack((frame, depth_display))
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(combined_display, f'Frame {idx + 1}/{len(image_files)}', 
                          (10, 30), font, 1, (255, 255, 255), 2)
                
                # Show the combined display
                cv2.imshow(window_name, combined_display)
                
                # Save frame to video
                if video_writer is not None:
                    video_writer.write(combined_display)
                
                # Brief pause between frames and check for quit
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break

        finally:
            cv2.destroyAllWindows()
            if video_writer is not None:
                video_writer.release()

        if all_points:
            points_array = np.array(all_points)
            colors_array = np.array(all_colors)
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_array)
            pcd.colors = o3d.utility.Vector3dVector(colors_array)
            
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            
            print(f"\nSaving point cloud to {pointcloud_path}")
            o3d.io.write_point_cloud(pointcloud_path, pcd)
            print(f"Saved point cloud with {len(pcd.points)} points")
            
            if self.pc_config['enabled']:
                self._visualize_point_cloud(pcd)

    def _visualize_point_cloud(self, pcd):
        """Visualize a point cloud using Open3D."""
        vis = o3d.visualization.Visualizer()
        vis.create_window("Point Cloud Viewer")
        
        vis.add_geometry(pcd)
        
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.set_front([0, 0, -1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, -1, 0])
        
        opt = vis.get_render_option()
        opt.point_size = self.pc_config['point_size']
        opt.background_color = self.pc_config.get('background_color', [0.1, 0.1, 0.1])
        
        print("\nPoint Cloud Visualization Controls:")
        print("- Left mouse button: Rotate")
        print("- Middle mouse button: Pan")
        print("- Right mouse button: Zoom")
        print("- '[' and ']': Change point size")
        print("- 'r': Reset view")
        print("- 'q': Close viewer")
        
        vis.run()
        vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description='Process image sequence for depth estimation and point cloud generation')
    parser.add_argument('--weight_path', type=str, required=True,
                      help='Path to model weights directory')
    parser.add_argument('--input_path', type=str, required=True,
                      help='Path to input image sequence directory')
    parser.add_argument('--config', type=str, default='config.json',
                      help='Path to configuration file')
    parser.add_argument('--image_format', type=str, default='jpg',
                      help='Input image format (jpg, png, etc.)')
    parser.add_argument('--no_cuda', action='store_true',
                      help='Disable CUDA if available')
    
    args = parser.parse_args()
    
    processor = SequenceProcessor(args.weight_path, args.config, not args.no_cuda)
    processor.process_sequence(args.input_path, args.image_format)

if __name__ == "__main__":
    main()