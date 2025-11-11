import argparse
import json
import time
from typing import List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np
import open3d as o3d
import requests

from lib.depth_point_cloud_utils import (
    DepthProcessor,
    PointCloudVisualizer,
    colorize_disparity,
)
from lib.point_cloud_preprocessing import PointCloudPreprocessor


CAP_ANY = getattr(cv2, "CAP_ANY", 0)
CAP_FFMPEG = getattr(cv2, "CAP_FFMPEG", None)
CAP_GSTREAMER = getattr(cv2, "CAP_GSTREAMER", None)
CAP_V4L2 = getattr(cv2, "CAP_V4L2", None)


class MjpegStreamReader:
    """Lightweight MJPEG reader for HTTP multipart streams."""

    def __init__(self, url: str, timeout: float = 5.0) -> None:
        self.url = url
        self.timeout = timeout
        self.session: Optional[requests.Session] = None
        self.response: Optional[requests.Response] = None
        self.boundary: Optional[bytes] = None
        self.buffer = b""

    def open(self) -> None:
        self.close()
        self.session = requests.Session()
        self.response = self.session.get(
            self.url,
            stream=True,
            timeout=self.timeout,
            headers={"User-Agent": "RTMonoDepth-MobileClient"},
        )
        self.response.raise_for_status()
        content_type = self.response.headers.get("Content-Type", "")
        if "boundary=" not in content_type:
            raise ValueError("Stream does not advertise multipart boundary")
        boundary = content_type.split("boundary=")[-1].strip().strip('"')
        if not boundary.startswith("--"):
            boundary = "--" + boundary
        self.boundary = boundary.encode()
        self.buffer = b""

    def read(self) -> Optional[np.ndarray]:
        if self.response is None or self.boundary is None:
            return None

        for chunk in self.response.iter_content(chunk_size=4096):
            if not chunk:
                continue
            self.buffer += chunk

            while True:
                boundary_index = self.buffer.find(self.boundary)
                if boundary_index == -1:
                    break

                part = self.buffer[:boundary_index]
                self.buffer = self.buffer[boundary_index + len(self.boundary):]

                header_end = part.find(b"\r\n\r\n")
                if header_end == -1:
                    continue

                frame_bytes = part[header_end + 4:]
                if not frame_bytes:
                    continue

                image = cv2.imdecode(
                    np.frombuffer(frame_bytes, dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if image is not None:
                    return image

        return None

    def close(self) -> None:
        if self.response is not None:
            self.response.close()
            self.response = None
        if self.session is not None:
            self.session.close()
            self.session = None
        self.boundary = None
        self.buffer = b""


class JpegSnapshotReader:
    """Simple poller for single-frame JPEG endpoints."""

    def __init__(self, url: str, timeout: float = 5.0) -> None:
        self.url = url
        self.timeout = timeout
        self.session: Optional[requests.Session] = None

    def open(self) -> None:
        self.close()
        self.session = requests.Session()

    def read(self) -> Optional[np.ndarray]:
        if self.session is None:
            return None

        try:
            response = self.session.get(
                self.url,
                stream=False,
                timeout=self.timeout,
                headers={"User-Agent": "RTMonoDepth-MobileClient"},
            )
            response.raise_for_status()
            frame = cv2.imdecode(
                np.frombuffer(response.content, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            return frame
        except requests.RequestException as exc:
            print(f"Snapshot fetch failed: {exc}")
            return None

    def close(self) -> None:
        if self.session is not None:
            self.session.close()
            self.session = None


class MobileCameraPointCloud:
    """Stream frames from an IP camera and visualize depth + point cloud."""

    def __init__(
        self,
        stream_url: str,
        weight_path: str,
        config_path: str,
        use_cuda: bool,
        reconnect_delay: float,
        frame_width: Optional[int],
        frame_height: Optional[int],
        extra_urls: Optional[List[str]],
        backend_choice: Optional[int],
    ) -> None:
        with open(config_path, "r", encoding="utf-8") as config_file:
            self.config = json.load(config_file)

        self.stream_url = stream_url
        self.reconnect_delay = max(reconnect_delay, 0.5)
        self.video_config = self.config["video"]
        self.model_config = self.config["model"]
        self.pc_config = self.config["point_cloud"]
        self.preprocess_config = self.config.get(
            "point_cloud_preprocessing", {}
        )

        self.override_width = frame_width or self.video_config.get("width")
        self.override_height = frame_height or self.video_config.get("height")
        self.extra_urls = extra_urls or []
        self.backend_choice = backend_choice

        self.depth_processor = DepthProcessor(
            weight_path,
            use_cuda,
            self.pc_config.get("min_depth", 0.1),
            self.pc_config.get("max_depth", 80.0),
            self.model_config.get("input_width", 640),
            self.model_config.get("input_height", 192),
        )
        self.visualizer = PointCloudVisualizer(self.config)
        self.preprocessor = PointCloudPreprocessor(self.preprocess_config)
        self.capture: Optional[cv2.VideoCapture] = None
        self.mjpeg_reader: Optional[MjpegStreamReader] = None
        self.snapshot_reader: Optional[JpegSnapshotReader] = None
        self.active_url: Optional[str] = None
        self.active_backend: Optional[int] = None

    def _candidate_urls(self) -> List[str]:
        """Generate a list of possible stream endpoints to try."""

        candidates = []
        base = self.stream_url.strip()
        if base:
            candidates.append(base)

        parts = urlsplit(base)
        if parts.scheme in {"http", "https"}:
            # Append ?action=stream variation if not already present
            if "?" not in base:
                candidates.append(f"{base}?action=stream")
                candidates.append(f"{base}?action=snapshot")

            # Ensure trailing /video path is considered (common for IP Webcam)
            if not parts.path.endswith("/video"):
                new_path = parts.path.rstrip("/") + "/video"
                candidates.append(urlunsplit(
                    (parts.scheme, parts.netloc, new_path, parts.query, parts.fragment)))
                if not parts.query:
                    candidates.append(urlunsplit(
                        (parts.scheme, parts.netloc, new_path, "action=stream", parts.fragment)))
                    candidates.append(urlunsplit(
                        (parts.scheme, parts.netloc, new_path, "action=snapshot", parts.fragment)))

            snapshot_variants = [
                "/shot.jpg",
                "/snapshot.jpg",
                "/jpeg",
            ]
            for variant in snapshot_variants:
                variant_path = variant if variant.startswith(
                    "/") else f"/{variant}"
                candidate = f"{parts.scheme}://{parts.netloc}{variant_path}"
                candidates.append(candidate)

        for extra in self.extra_urls:
            if extra:
                candidates.append(extra.strip())

        # Remove duplicates while preserving order
        unique = []
        seen = set()
        for url in candidates:
            if url and url not in seen:
                seen.add(url)
                unique.append(url)
        return unique

    def _backend_sequence(self) -> List[Optional[int]]:
        if self.backend_choice is not None:
            return [self.backend_choice]
        # Try OpenCV default first, then FFMPEG (helps with MJPEG streams)
        sequence = [CAP_ANY]
        if CAP_FFMPEG is not None:
            sequence.append(CAP_FFMPEG)
        return sequence

    def _try_mjpeg(self, url: str) -> bool:
        try:
            print(f"Attempting MJPEG fallback for {url} ...")
            reader = MjpegStreamReader(url)
            reader.open()
            self.mjpeg_reader = reader
            self.active_url = f"{url} [MJPEG]"
            self.active_backend = None
            print(f"Connected to {url} via MJPEG fallback")
            return True
        except Exception as exc:
            print(f"MJPEG fallback failed for {url}: {exc}")
            if self.mjpeg_reader is not None:
                self.mjpeg_reader.close()
                self.mjpeg_reader = None
            return False

    def _try_snapshot(self, url: str) -> bool:
        try:
            print(f"Attempting snapshot fallback for {url} ...")
            reader = JpegSnapshotReader(url)
            reader.open()
            frame = reader.read()
            if frame is None:
                raise RuntimeError("Snapshot endpoint returned no data")
            self.snapshot_reader = reader
            self.active_url = f"{url} [SNAPSHOT]"
            self.active_backend = None
            print(f"Connected to {url} via snapshot polling")
            return True
        except Exception as exc:
            print(f"Snapshot fallback failed for {url}: {exc}")
            if self.snapshot_reader is not None:
                self.snapshot_reader.close()
                self.snapshot_reader = None
            return False

    def _connect(self) -> bool:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if self.mjpeg_reader is not None:
            self.mjpeg_reader.close()
            self.mjpeg_reader = None
        if self.snapshot_reader is not None:
            self.snapshot_reader.close()
            self.snapshot_reader = None

        for url in self._candidate_urls():
            for backend in self._backend_sequence():
                backend_name = (
                    "CAP_ANY" if backend in (None, CAP_ANY)
                    else "CAP_FFMPEG" if backend == CAP_FFMPEG
                    else "CAP_GSTREAMER" if backend == CAP_GSTREAMER
                    else "CAP_V4L2" if backend == CAP_V4L2
                    else f"backend {backend}"
                )
                print(f"Connecting to {url} using {backend_name} ...")
                capture = cv2.VideoCapture(
                    url, backend if backend is not None else CAP_ANY)
                if capture.isOpened():
                    if self.override_width:
                        capture.set(cv2.CAP_PROP_FRAME_WIDTH,
                                    int(self.override_width))
                    if self.override_height:
                        capture.set(cv2.CAP_PROP_FRAME_HEIGHT,
                                    int(self.override_height))
                    self.capture = capture
                    self.active_url = url
                    self.active_backend = backend
                    print(f"Connected to {url} ({backend_name})")
                    return True
                capture.release()
                time.sleep(0.2)

            if self._try_mjpeg(url):
                return True

            if self._try_snapshot(url):
                return True

        print("Failed to open stream with provided URLs/backends; will retry")
        self.capture = None
        self.active_url = None
        self.active_backend = None
        return False

    def _read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.capture is not None:
            ret, frame = self.capture.read()
            if not ret or frame is None:
                print("Stream read failed; attempting to reconnect")
                self.capture.release()
                self.capture = None
                return False, None
            return True, frame

        if self.mjpeg_reader is not None:
            frame = self.mjpeg_reader.read()
            if frame is None:
                print("MJPEG stream stalled; attempting to reconnect")
                self.mjpeg_reader.close()
                self.mjpeg_reader = None
                return False, None
            if self.override_width and self.override_height:
                frame = cv2.resize(
                    frame,
                    (int(self.override_width), int(self.override_height)),
                    interpolation=cv2.INTER_LINEAR,
                )
            return True, frame

        if self.snapshot_reader is not None:
            frame = self.snapshot_reader.read()
            if frame is None:
                print("Snapshot stream stalled; attempting to reconnect")
                self.snapshot_reader.close()
                self.snapshot_reader = None
                return False, None
            if self.override_width and self.override_height:
                frame = cv2.resize(
                    frame,
                    (int(self.override_width), int(self.override_height)),
                    interpolation=cv2.INTER_LINEAR,
                )
            return True, frame

        return False, None

    def _process_point_cloud(self, color_frame: np.ndarray, depth_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
            print(
                f"Nearest obstacle cluster {nearest['label']} at {nearest['distance']:.2f} m"
            )

        return (
            np.asarray(processed_pcd.points),
            np.asarray(processed_pcd.colors),
        )

    def run(self) -> None:
        try:
            while True:
                has_capture = self.capture is not None and self.capture.isOpened()
                has_mjpeg = self.mjpeg_reader is not None
                has_snapshot = self.snapshot_reader is not None
                if not (has_capture or has_mjpeg or has_snapshot):
                    if not self._connect():
                        time.sleep(self.reconnect_delay)
                        continue

                ok, frame = self._read_frame()
                if not ok:
                    time.sleep(self.reconnect_delay)
                    continue

                depth_map, resized_frame, disp_map = self.depth_processor.estimate_depth(
                    frame)

                depth_colormap = colorize_disparity(disp_map)

                if self.pc_config.get("enabled", True):
                    points, colors = self._process_point_cloud(
                        resized_frame, depth_map
                    )
                else:
                    points = np.empty((0, 3), dtype=np.float32)
                    colors = np.empty((0, 3), dtype=np.float32)

                if len(points) > 0:
                    depth_min = float(points[:, 2].min())
                    depth_max = float(points[:, 2].max())
                    print(
                        f"Points generated: {len(points)}, Depth range: {depth_min:.2f}m to {depth_max:.2f}m"
                    )

                self.visualizer.update(points, colors)

                stacked = np.hstack(
                    (
                        frame,
                        cv2.resize(depth_colormap,
                                   (frame.shape[1], frame.shape[0])),
                    )
                )
                cv2.imshow("Mobile RGB | Depth", stacked)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    self.visualizer.reset_view()
        finally:
            cv2.destroyAllWindows()
            if self.capture is not None:
                self.capture.release()
            if self.mjpeg_reader is not None:
                self.mjpeg_reader.close()
            if self.snapshot_reader is not None:
                self.snapshot_reader.close()
            self.visualizer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream mobile camera feed into RT-MonoDepth pipeline",
    )
    parser.add_argument("--url", required=True, help="HTTP or RTSP stream URL")
    parser.add_argument("--weight_path", required=True,
                        help="RT-MonoDepth weights directory")
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to configuration file (camera intrinsics, visualization)",
    )
    parser.add_argument(
        "--reconnect_delay",
        type=float,
        default=2.0,
        help="Seconds to wait before retrying the stream after a drop",
    )
    parser.add_argument(
        "--frame_width",
        type=int,
        default=0,
        help="Optional override for captured frame width",
    )
    parser.add_argument(
        "--frame_height",
        type=int,
        default=0,
        help="Optional override for captured frame height",
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Force depth inference on CPU",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "ffmpeg", "gstreamer", "v4l2"],
        default="auto",
        help="Preferred OpenCV capture backend",
    )
    parser.add_argument(
        "--extra_url",
        action="append",
        default=[],
        help="Additional URL variant to try if the main stream fails (can be repeated)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend_map = {
        "auto": None,
        "ffmpeg": CAP_FFMPEG,
        "gstreamer": CAP_GSTREAMER,
        "v4l2": CAP_V4L2,
    }
    runner = MobileCameraPointCloud(
        stream_url=args.url,
        weight_path=args.weight_path,
        config_path=args.config,
        use_cuda=not args.no_cuda,
        reconnect_delay=args.reconnect_delay,
        frame_width=args.frame_width if args.frame_width > 0 else None,
        frame_height=args.frame_height if args.frame_height > 0 else None,
        extra_urls=args.extra_url,
        backend_choice=backend_map.get(args.backend, None),
    )
    runner.run()


if __name__ == "__main__":
    main()
