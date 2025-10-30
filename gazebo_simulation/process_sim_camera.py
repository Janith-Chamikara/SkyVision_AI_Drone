import argparse
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Event, Lock
from typing import Optional

import cv2
import numpy as np
from gz.msgs10.image_pb2 import Image
from gz.transport13 import Node

# Ensure the project root is importable when this script is launched directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from process_sequence import SequenceProcessor  # noqa: E402


# Debian packages for Gazebo strip the generated enum helpers, so we keep a
# manual map of the pixel format codes we care about. Values follow
# ignition::msgs::PixelFormatType.
PIXEL_FORMAT_HANDLERS = {
    3: ("RGB_INT8", 3, lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2BGR)),
    4: ("RGBA_INT8", 4, lambda img: cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)),
    5: ("BGRA_INT8", 4, lambda img: cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)),
    8: ("BGR_INT8", 3, lambda img: img),
}
PNG_EXTENSION = "png"


@dataclass
class CaptureConfig:
    topic: str
    output_root: Path
    frame_limit: int
    preview: bool
    weight_path: Path
    config_path: Path
    use_cuda: bool


class SimulationSequenceRunner:
    def __init__(self, capture_cfg: CaptureConfig):
        self.cfg = capture_cfg
        self.node = Node()
        self.shutdown_event = Event()
        self.frame_lock = Lock()
        self.frame_counter = 0
        self.sequence_path = self._prepare_sequence_dir()
        self.latest_image: Optional[np.ndarray] = None
        self.sequence_processor: Optional[SequenceProcessor] = None

    def _prepare_sequence_dir(self) -> Path:
        session_dir = self.cfg.output_root / datetime.now().strftime("sim_%Y%m%d_%H%M%S")
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def _subscribe(self) -> None:
        if not self.node.subscribe(Image, self.cfg.topic, self._image_callback):
            raise RuntimeError(
                f"Failed to subscribe to Gazebo topic '{self.cfg.topic}'")
        print(f"Subscribed to Gazebo camera topic: {self.cfg.topic}")
        print(f"Frames will be stored in: {self.sequence_path}")

    def _convert_message(self, msg: Image) -> np.ndarray:
        pixel_format = msg.pixel_format_type
        handler = PIXEL_FORMAT_HANDLERS.get(pixel_format)
        if handler is None:
            known_formats = ", ".join(
                f"{value} ({fmt_name})" for value, (fmt_name, _, __) in PIXEL_FORMAT_HANDLERS.items()
            ) or "none"
            raise ValueError(
                "Unsupported pixel format "
                f"{pixel_format}; expected one of: {known_formats}"
            )

        _, channels, converter = handler

        np_data = np.frombuffer(msg.data, dtype=np.uint8)
        expected_size = msg.width * msg.height * channels
        if np_data.size != expected_size:
            raise ValueError(
                f"Unexpected image buffer size {np_data.size}, expected {expected_size}"
            )

        image = np_data.reshape((msg.height, msg.width, channels))
        converted = converter(image)
        if converted.ndim == 2:
            converted = cv2.cvtColor(converted, cv2.COLOR_GRAY2BGR)
        return converted

    def _image_callback(self, msg: Image) -> None:
        if self.shutdown_event.is_set():
            return

        try:
            image_bgr = self._convert_message(msg)
        except ValueError as exc:
            print(f"Skipping frame: {exc}")
            return

        with self.frame_lock:
            self.frame_counter += 1
            frame_id = self.frame_counter

        frame_path = self.sequence_path / f"{frame_id:06d}.{PNG_EXTENSION}"
        if not cv2.imwrite(str(frame_path), image_bgr):
            print(f"Warning: failed to write frame {frame_path}")
            return

        self.latest_image = image_bgr
        if self.cfg.preview:
            cv2.imshow("Gazebo Camera", image_bgr)
            cv2.waitKey(1)

        if self.cfg.frame_limit > 0 and frame_id >= self.cfg.frame_limit:
            print(
                f"Reached frame limit ({self.cfg.frame_limit}); stopping capture")
            self.shutdown_event.set()

    def _install_signal_handlers(self) -> None:
        def handle_signal(signum, _frame):
            print(f"Received signal {signum}; finishing capture...")
            self.shutdown_event.set()

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

    def _finalize(self) -> None:
        if self.cfg.preview:
            cv2.destroyAllWindows()

        total_frames = self.frame_counter
        if total_frames == 0:
            print("No frames captured; skipping depth processing")
            return

        print(f"Captured {total_frames} frame(s); running depth pipeline")
        self.sequence_processor = SequenceProcessor(
            str(self.cfg.weight_path), str(
                self.cfg.config_path), self.cfg.use_cuda
        )
        self.sequence_processor.process_sequence(
            str(self.sequence_path), PNG_EXTENSION)

    def spin(self) -> None:
        self._install_signal_handlers()
        self._subscribe()
        print("Capturing Gazebo frames. Press Ctrl+C to stop.")

        try:
            while not self.shutdown_event.is_set():
                time.sleep(0.1)
        finally:
            self.shutdown_event.set()
            self._finalize()


def parse_args() -> CaptureConfig:
    parser = argparse.ArgumentParser(
        description="Capture Gazebo camera frames and run the depth/point-cloud pipeline"
    )
    parser.add_argument("--topic", default="/camera/image",
                        help="Gazebo image topic")
    parser.add_argument(
        "--output_root",
        default="simulation_frames",
        help="Directory where frame batches will be saved",
    )
    parser.add_argument(
        "--frame_limit",
        type=int,
        default=0,
        help="Max frames to capture before processing (0 captures until interrupted)",
    )
    parser.add_argument("--weight_path", required=True,
                        help="Path to RT-MonoDepth weights")
    parser.add_argument(
        "--config",
        default="config_gazebo.json",
        help="Path to camera/pipeline configuration file",
    )
    parser.add_argument("--no_preview", action="store_true",
                        help="Disable OpenCV preview window")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Force inference on CPU")

    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    capture_cfg = CaptureConfig(
        topic=args.topic,
        output_root=output_root,
        frame_limit=max(0, args.frame_limit),
        preview=not args.no_preview,
        weight_path=Path(args.weight_path).expanduser().resolve(),
        config_path=Path(args.config).expanduser().resolve(),
        use_cuda=not args.no_cuda,
    )
    return capture_cfg


def main() -> None:
    capture_cfg = parse_args()
    runner = SimulationSequenceRunner(capture_cfg)
    runner.spin()


if __name__ == "__main__":
    main()
