import argparse
import json
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import open3d as o3d

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
except ImportError as exc:  # pragma: no cover - requires ROS environment
    raise SystemExit(
        "ROS dependencies are missing. Install rclpy and cv_bridge from your ROS distribution."
    ) from exc

from lib.depth_point_cloud_utils import (
    DepthProcessor,
    PointCloudVisualizer,
    colorize_disparity,
)
from lib.point_cloud_preprocessing import PointCloudPreprocessor


class ROSCameraPointCloud(Node):
    """Subscribe to a ROS 2 image topic and feed frames through the SkyVision pipeline."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("ros_camera_pointcloud")

        with open(args.config, "r", encoding="utf-8") as cfg_file:
            self.config = json.load(cfg_file)

        self.video_config = self.config["video"]
        self.model_config = self.config["model"]
        self.pc_config = self.config["point_cloud"]
        self.preprocess_config = self.config.get(
            "point_cloud_preprocessing", {}
        )

        self.override_width: Optional[int] = args.frame_width
        self.override_height: Optional[int] = args.frame_height
        self.min_interval = max(0.0, args.min_interval)
        self.window_name = "ROS RGB | Depth"

        self.depth_processor = DepthProcessor(
            args.weight_path,
            use_cuda=not args.no_cuda,
            min_depth=self.pc_config.get("min_depth", 0.1),
            max_depth=self.pc_config.get("max_depth", 80.0),
            input_width=self.model_config.get("input_width", 640),
            input_height=self.model_config.get("input_height", 192),
        )
        self.visualizer = PointCloudVisualizer(self.config)
        self.preprocessor = PointCloudPreprocessor(self.preprocess_config)

        self.bridge = CvBridge()
        self.processing_lock = threading.Lock()
        self.last_frame_time = 0.0
        self.shutdown_requested = False

        qos = QoSProfile(
            depth=args.queue_size,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.subscription = self.create_subscription(
            Image,
            args.topic,
            self._image_callback,
            qos,
        )

        self.get_logger().info(
            f"Subscribed to {args.topic} with queue size {args.queue_size}"
        )
        self.get_logger().info("Press 'q' in the OpenCV window to exit, 'r' to reset view.")

    # ---------------------------------------------------------------------
    # ROS Callback and Core Processing
    # ---------------------------------------------------------------------
    def _image_callback(self, msg: Image) -> None:
        if self.shutdown_requested:
            return

        now = time.monotonic()
        if self.min_interval > 0.0 and now - self.last_frame_time < self.min_interval:
            return

        if not self.processing_lock.acquire(blocking=False):
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:  # pragma: no cover - relies on ROS runtime
            self.get_logger().warning(f"Failed to convert image: {exc}")
            self.processing_lock.release()
            return

        try:
            if self.override_width and self.override_height:
                frame = cv2.resize(
                    frame,
                    (int(self.override_width), int(self.override_height)),
                    interpolation=cv2.INTER_LINEAR,
                )

            depth_map, resized_frame, disp_map = self.depth_processor.estimate_depth(
                frame
            )
            depth_colormap = colorize_disparity(disp_map)

            if self.pc_config.get("enabled", True):
                points, colors = self._process_point_cloud(
                    resized_frame, depth_map)
            else:
                points = np.empty((0, 3), dtype=np.float32)
                colors = np.empty((0, 3), dtype=np.float32)

            if len(points) > 0:
                depth_min = float(points[:, 2].min())
                depth_max = float(points[:, 2].max())
                self.get_logger().debug(
                    f"Points generated: {len(points)} | Depth range {depth_min:.2f}m - {depth_max:.2f}m"
                )

            self.visualizer.update(points, colors)

            max_display_width = 1920
            original_height, original_width = frame.shape[:2]
            combined_width = original_width * 2

            if combined_width > max_display_width:
                scale = max_display_width / float(combined_width)
                display_width = int(original_width * scale)
                display_height = int(original_height * scale)
            else:
                display_width = original_width
                display_height = original_height

            depth_resized = cv2.resize(
                depth_colormap, (original_width, original_height))
            frame_display = cv2.resize(frame, (display_width, display_height))
            depth_display = cv2.resize(
                depth_resized, (display_width, display_height))
            stacked = np.hstack((frame_display, depth_display))
            cv2.imshow(self.window_name, stacked)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.get_logger().info("'q' pressed – shutting down node.")
                self.shutdown_requested = True
                rclpy.shutdown()
            elif key == ord("r"):
                self.visualizer.reset_view()

            self.last_frame_time = now
        except Exception as exc:  # pragma: no cover - runtime protection
            self.get_logger().error(f"Processing failure: {exc}")
        finally:
            self.processing_lock.release()

    # ------------------------------------------------------------------
    def _process_point_cloud(
        self, color_frame: np.ndarray, depth_map: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        points, colors = self.visualizer.create_point_cloud(
            color_frame, depth_map)
        if len(points) == 0:
            return points, colors

        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = o3d.utility.Vector3dVector(points)
        temp_pcd.colors = o3d.utility.Vector3dVector(colors)

        processed_pcd, report = self.preprocessor.process(temp_pcd)
        nearest = report.get("nearest_obstacle", {})
        if nearest.get("distance") is not None:
            label = nearest.get("label", "?")
            distance = nearest["distance"]
            self.get_logger().info(
                f"Nearest obstacle cluster {label} at {distance:.2f} m"
            )

        return (
            np.asarray(processed_pcd.points),
            np.asarray(processed_pcd.colors),
        )

    # ------------------------------------------------------------------
    def close(self) -> None:
        cv2.destroyAllWindows()
        self.visualizer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feed ROS 2 camera frames into the RT-MonoDepth point cloud pipeline",
    )
    parser.add_argument(
        "--topic",
        default="/camera_out",
        help="ROS 2 image topic to subscribe to",
    )
    parser.add_argument(
        "--weight_path",
        required=True,
        help="Path to RT-MonoDepth weights directory",
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Pipeline configuration file (intrinsics, visualization)",
    )
    parser.add_argument(
        "--queue_size",
        type=int,
        default=1,
        help="ROS 2 subscription queue size",
    )
    parser.add_argument(
        "--frame_width",
        type=int,
        default=0,
        help="Optional frame width override before inference",
    )
    parser.add_argument(
        "--frame_height",
        type=int,
        default=0,
        help="Optional frame height override before inference",
    )
    parser.add_argument(
        "--min_interval",
        type=float,
        default=0.0,
        help="Minimum seconds between processed frames (throttle high-rate topics)",
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Force depth inference on CPU",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frame_width and args.frame_width < 0:
        raise ValueError("frame_width must be positive when provided")
    if args.frame_height and args.frame_height < 0:
        raise ValueError("frame_height must be positive when provided")

    rclpy.init()
    node = ROSCameraPointCloud(args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received – shutting down.")
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
