import cv2
import json
from lib.video_utils import VideoSource
from lib.depth_point_cloud_utils import DepthProcessor, PointCloudVisualizer
import numpy as np

class DepthToPointCloud:
    def __init__(self, weight_path, config_path="config.json", use_cuda=True):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        self.depth_processor = DepthProcessor(weight_path, use_cuda)
        self.visualizer = PointCloudVisualizer(self.config)
        
        self.video_config = self.config['video']
        self.model_config = self.config['model']
        self.pc_config = self.config['point_cloud']

    def process_frame(self, frame):
        """Process a single frame to generate depth map and point cloud."""
        depth_map, resized_frame = self.depth_processor.estimate_depth(frame)
        return depth_map, resized_frame

    def run(self):
        """Main processing loop."""
        try:
            source = 0 if self.video_config['source'] == 'WEB_CAM' else self.video_config['source']
            with VideoSource(source, 
                           self.video_config['width'],
                           self.video_config['height']) as video:
                while True:
                    ret, frame = video.get_frame()
                    if not ret:
                        break

                    depth_map, resized_frame = self.process_frame(frame)
                    
                    depth_colormap = cv2.applyColorMap(
                        cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U),
                        cv2.COLORMAP_MAGMA
                    )

                    if self.pc_config['enabled']:
                        points, colors = self.visualizer.create_point_cloud(
                            resized_frame,
                            depth_map * self.pc_config['depth_scale']
                        )
                    
                    if len(points) > 0:
                        print(f"Points generated: {len(points)}, Depth range: {points[:,2].min():.2f} to {points[:,2].max():.2f}")
                    
                    self.visualizer.update(points, colors)

                    cv2.imshow('RT-MonoDepth - RGB | Depth', 
                             np.hstack((frame, cv2.resize(depth_colormap, (frame.shape[1], frame.shape[0]))))
                    )

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        self.visualizer.reset_view()

        finally:
            cv2.destroyAllWindows()
            self.visualizer.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='RT-MonoDepth with Point Cloud visualization')
    parser.add_argument('--weight_path', type=str, required=True,
                        help='Path to model weights directory')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to configuration file')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA if available')
    
    args = parser.parse_args()
    
    processor = DepthToPointCloud(args.weight_path, args.config, not args.no_cuda)
    processor.run()