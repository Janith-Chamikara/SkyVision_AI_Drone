import cv2
import json
import os
import glob
import numpy as np
from lib.depth_point_cloud_utils import (
    DepthProcessor,
    build_pixel_grid,
    compute_scaled_intrinsics,
    depth_to_points
)
from lib.point_cloud_preprocessing import PointCloudPreprocessor
import argparse
import open3d as o3d


class SequenceProcessor:
    def __init__(self, weight_path, config_path="config.json", use_cuda=True):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.video_config = self.config['video']
        self.model_config = self.config['model']
        self.pc_config = self.config['point_cloud']
        self.preprocess_config = self.config.get(
            'point_cloud_preprocessing', {})

        self.min_depth = self.pc_config.get('min_depth', 0.1)
        self.max_depth = self.pc_config.get('max_depth', 80.0)
        self.apply_extrinsics = self.pc_config.get('apply_extrinsics', False)

        self.depth_processor = DepthProcessor(
            weight_path,
            use_cuda,
            self.min_depth,
            self.max_depth,
            self.model_config.get('input_width', 640),
            self.model_config.get('input_height', 192)
        )
        self.preprocessor = PointCloudPreprocessor(self.preprocess_config)

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.video_dir = os.path.join(self.base_dir, 'depth_videos')
        self.depth_maps_root = os.path.join(self.base_dir, 'depth_maps')
        self.pointcloud_dir = os.path.join(self.base_dir, 'point_clouds')
        os.makedirs(self.video_dir, exist_ok=True)
        os.makedirs(self.depth_maps_root, exist_ok=True)
        os.makedirs(self.pointcloud_dir, exist_ok=True)

        self._camera_to_body = self._load_camera_to_body()
        self._pixel_grid = None
        self._intrinsics = None
        self._depth_shape = None
        self._original_shape = None

    def _load_camera_to_body(self):
        transform = self.pc_config.get('camera_to_body')
        if transform is None:
            return np.eye(4, dtype=np.float32)

        try:
            transform = np.asarray(transform, dtype=np.float32)
            if transform.size != 16:
                raise ValueError
            return transform.reshape(4, 4)
        except ValueError:
            print("Warning: Invalid camera_to_body transform; using identity matrix.")
            return np.eye(4, dtype=np.float32)

    def _ensure_geometry_cache(self, original_shape, depth_shape):
        if (self._depth_shape == depth_shape) and (self._original_shape == original_shape):
            return

        self._intrinsics = compute_scaled_intrinsics(
            self.video_config, original_shape, depth_shape)
        self._pixel_grid = build_pixel_grid(depth_shape)
        self._depth_shape = depth_shape
        self._original_shape = original_shape

    def _frame_to_points(self, depth_map, color_bgr):
        if self._intrinsics is None or self._pixel_grid is None:
            raise RuntimeError(
                "Projection cache not prepared before point cloud conversion")

        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        transform = self._camera_to_body if self.apply_extrinsics else None

        points, colors = depth_to_points(
            depth_map,
            color_rgb,
            self._intrinsics,
            self.min_depth,
            self.max_depth,
            pixel_grid=self._pixel_grid,
            transform=transform
        )

        return points, colors

    def process_sequence(self, input_path, image_format='jpg'):
        """Two-phase pipeline: 1) save per-frame depth maps, 2) build point cloud from saved depth maps."""

        input_dir_name = os.path.basename(os.path.normpath(input_path))
        video_path = os.path.join(
            self.video_dir, f'{input_dir_name}_depth.mp4')
        pointcloud_path = os.path.join(
            self.pointcloud_dir, f'{input_dir_name}_pointcloud.ply')
        image_files = sorted(
            glob.glob(os.path.join(input_path, f'*.{image_format}')))
        depth_dir = os.path.join(self.depth_maps_root, input_dir_name)
        os.makedirs(depth_dir, exist_ok=True)

        if not image_files:
            print(f"No {image_format} files found in {input_path}")
            return

        print(f"Found {len(image_files)} images to process")

        # ----------------------------
        # Phase 1: Predict and save depth
        # ----------------------------
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

            print("\nPhase 1/2: Estimating depth and saving to disk...")
            print("Press 'Q' to stop early")

            for idx, image_file in enumerate(image_files):
                frame = cv2.imread(image_file)
                if frame is None:
                    print(f"Failed to read image: {image_file}")
                    continue

                depth_map, _ = self.depth_processor.estimate_depth(frame)

                self._ensure_geometry_cache(frame.shape[:2], depth_map.shape)

                depth_colormap = cv2.applyColorMap(
                    cv2.normalize(depth_map, None, 0, 255,
                                  cv2.NORM_MINMAX, cv2.CV_8U),
                    cv2.COLORMAP_MAGMA
                )

                depth_display = cv2.resize(
                    depth_colormap, (frame.shape[1], frame.shape[0]))

                # Save depth artifacts
                base = os.path.splitext(os.path.basename(image_file))[0]
                npy_path = os.path.join(depth_dir, f"{base}.npy")
                png_path = os.path.join(depth_dir, f"{base}_depth.png")
                try:
                    # Save the raw depth for accurate downstream use
                    np.save(npy_path, depth_map.astype(np.float32))
                    # Save a preview image for quick inspection
                    cv2.imwrite(png_path, depth_colormap)
                except Exception as e:
                    print(
                        f"Warning: Failed to save depth for {image_file}: {e}")

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

        # ----------------------------
        # Phase 2: Build point cloud from saved depth maps
        # ----------------------------
        if self.pc_config['enabled']:
            print("\nPhase 2/2: Building point cloud from saved depth maps...")
            depth_npy_files = sorted(
                glob.glob(os.path.join(depth_dir, '*.npy')))
            if not depth_npy_files:
                print(
                    f"No saved depth maps found in {depth_dir}. Nothing to build.")
                return

            base_to_img = {
                os.path.splitext(os.path.basename(p))[0]: p for p in image_files
            }

            all_points = []
            all_colors = []

            for idx, npy_file in enumerate(depth_npy_files):
                base = os.path.splitext(os.path.basename(npy_file))[0]
                img_path = base_to_img.get(base)
                if img_path is None:
                    continue

                depth_map = np.load(npy_file).astype(np.float32)
                if depth_map.ndim != 2:
                    print(f"Skipping {npy_file}: depth map must be 2D")
                    continue

                color_bgr = cv2.imread(img_path)
                if color_bgr is None:
                    print(f"Skipping {img_path}: unable to load RGB frame")
                    continue

                self._ensure_geometry_cache(
                    color_bgr.shape[:2], depth_map.shape)

                color_resized = cv2.resize(
                    color_bgr,
                    (depth_map.shape[1], depth_map.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )

                frame_points, frame_colors = self._frame_to_points(
                    depth_map, color_resized)

                if frame_points.size == 0:
                    continue

                all_points.append(frame_points)
                all_colors.append(frame_colors)

                print(
                    f"Depth {idx + 1}/{len(depth_npy_files)} - Valid points: {len(frame_points)}"
                )

            if all_points:
                points_array = np.concatenate(all_points, axis=0)
                colors_array = np.concatenate(all_colors, axis=0)

                raw_pcd = o3d.geometry.PointCloud()
                raw_pcd.points = o3d.utility.Vector3dVector(points_array)
                raw_pcd.colors = o3d.utility.Vector3dVector(colors_array)

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_array.copy())
                pcd.colors = o3d.utility.Vector3dVector(colors_array.copy())

                voxel_size = self.pc_config.get('voxel_size', 0.0)
                if voxel_size and voxel_size > 0:
                    pcd = pcd.voxel_down_sample(float(voxel_size))

                outlier_cfg = self.pc_config.get('statistical_outlier', {})
                if not isinstance(outlier_cfg, dict):
                    outlier_cfg = {}
                if outlier_cfg.get('enabled', True):
                    nb_neighbors = int(outlier_cfg.get('nb_neighbors', 20))
                    std_ratio = float(outlier_cfg.get('std_ratio', 2.0))
                    pcd, _ = pcd.remove_statistical_outlier(
                        nb_neighbors=nb_neighbors, std_ratio=std_ratio)

                print("\nPreviewing raw point cloud before preprocessing...")
                self._visualize_point_cloud(raw_pcd)

                processed_pcd, preprocess_report = self.preprocessor.process(
                    pcd)

                print(f"\nSaving point cloud to {pointcloud_path}")
                o3d.io.write_point_cloud(pointcloud_path, processed_pcd)
                print(
                    f"Saved point cloud with {len(processed_pcd.points)} points")

                if preprocess_report.get('steps'):
                    analysis_path = os.path.join(
                        self.pointcloud_dir, f'{input_dir_name}_analysis.json')
                    with open(analysis_path, 'w', encoding='utf-8') as analysis_file:
                        json.dump(preprocess_report, analysis_file, indent=2)
                    print(f"Preprocessing report saved to {analysis_path}")

                self._visualize_point_cloud(processed_pcd)
            else:
                print("No valid 3D points were generated from the saved depth maps.")

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
        opt.background_color = self.pc_config.get(
            'background_color', [0.1, 0.1, 0.1])

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
    parser = argparse.ArgumentParser(
        description='Process image sequence for depth estimation and point cloud generation')
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

    processor = SequenceProcessor(
        args.weight_path, args.config, not args.no_cuda)
    processor.process_sequence(args.input_path, args.image_format)


if __name__ == "__main__":
    main()
