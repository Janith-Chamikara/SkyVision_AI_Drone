"""
Autonomous Drone Navigation System
Integrates depth estimation, point cloud processing, obstacle avoidance, and motor control
Optimized for Jetson real-time execution at 30 Hz
"""

import argparse
import json
import cv2
import numpy as np
import time
import threading
from collections import deque
from typing import Optional, Dict

# Import our modules
from lib.depth_point_cloud_utils import DepthProcessor
from lib.processing.point_cloud_processor import PointCloudProcessor
from lib.navigation.occupancy_grid import OccupancyGrid3D
from lib.navigation.vfh_plus import VFHPlusNavigator
from lib.control.motor_controller import QuadcopterMotorController, SimulatedMotorController
from lib.video_utils import VideoSource


class AutonomousDroneSystem:
    """
    Complete autonomous navigation system with three-threaded architecture:
    
    Thread 1 (Camera): Captures frames and generates point clouds (30-60 Hz)
    Thread 2 (Navigation): Processes point clouds and computes commands (30 Hz)
    Thread 3 (Motors): Sends PWM commands to flight controller (50 Hz)
    """
    
    def __init__(self, weight_path: str, config_path: str, source, 
                 use_cuda: bool = True, enable_motors: bool = False):
        """
        Initialize autonomous navigation system.
        
        Args:
            weight_path: Path to RTMonoDepth weights
            config_path: Path to configuration JSON
            source: Video source (webcam index, RTSP URL, or file path)
            use_cuda: Enable CUDA acceleration
            enable_motors: Enable actual motor control (vs simulation)
        """
        print("\n" + "="*70)
        print(" Autonomous Drone Navigation System - Jetson Optimized")
        print("="*70)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Video source
        self.source = source
        self.video_source = None
        
        # Thread 1: Depth estimation
        print("\n[1/5] Initializing depth processor...")
        self.depth_processor = DepthProcessor(
            weight_path,
            use_cuda,
            self.config['point_cloud']['min_depth'],
            self.config['point_cloud']['max_depth'],
            self.config['model']['input_width'],
            self.config['model']['input_height']
        )
        
        # Thread 1: Point cloud processing
        print("[2/5] Initializing point cloud processor...")
        self.pc_processor = PointCloudProcessor(
            self.config.get('point_cloud_processing', {})
        )
        
        # Thread 2: Occupancy grid
        print("[3/5] Initializing occupancy grid...")
        self.occupancy_grid = OccupancyGrid3D(
            self.config.get('occupancy_grid', {})
        )
        
        # Thread 2: VFH+ navigator
        print("[4/5] Initializing VFH+ navigator...")
        self.navigator = VFHPlusNavigator(
            self.config['navigation']
        )
        
        # Thread 3: Motor controller
        print("[5/5] Initializing motor controller...")
        self.enable_motors = enable_motors
        if enable_motors:
            self.motor_controller = QuadcopterMotorController(
                self.config.get('motor_control', {})
            )
        else:
            self.motor_controller = SimulatedMotorController(
                self.config.get('motor_control', {})
            )
        
        # Threading infrastructure
        self.running = False
        self.camera_thread = None
        self.navigation_thread = None
        
        # Shared data (with locks)
        self.latest_frame = None
        self.latest_points = None
        self.latest_colors = None
        self.latest_nav_command = None
        self.frame_lock = threading.Lock()
        self.points_lock = threading.Lock()
        self.command_lock = threading.Lock()
        
        # Performance tracking
        self.camera_fps = deque(maxlen=30)
        self.navigation_fps = deque(maxlen=30)
        self.frame_count = 0
        self.nav_count = 0
        
        # Control parameters
        self.target_nav_rate = self.config['navigation'].get('control_frequency', 30)
        self.nav_period = 1.0 / self.target_nav_rate
        
        # Safety
        self.emergency_stop_flag = False
        
        print("\n" + "="*70)
        print(" System initialized successfully!")
        print(f" Mode: {'LIVE HARDWARE' if enable_motors else 'SIMULATION'}")
        print(f" Target navigation rate: {self.target_nav_rate} Hz")
        print("="*70 + "\n")
    
    def initialize_video_source(self) -> bool:
        """Initialize video capture."""
        try:
            # Handle webcam index
            source = self.source
            if isinstance(self.source, str) and self.source.isdigit():
                source = int(self.source)
            
            self.video_source = VideoSource(
                source,
                self.config['video']['width'],
                self.config['video']['height']
            )
            
            print(f"✓ Video source initialized: {source}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to initialize video source: {e}")
            return False
    
    def start(self) -> bool:
        """Start all threads."""
        # Initialize video
        if not self.initialize_video_source():
            return False
        
        # Start motor controller
        if not self.motor_controller.start():
            if self.enable_motors:
                print("Failed to start motor controller")
                return False
        
        self.running = True
        
        # Start camera thread (Thread 1)
        self.camera_thread = threading.Thread(
            target=self._camera_loop,
            name="CameraThread",
            daemon=True
        )
        self.camera_thread.start()
        
        # Start navigation thread (Thread 2)
        self.navigation_thread = threading.Thread(
            target=self._navigation_loop,
            name="NavigationThread",
            daemon=True
        )
        self.navigation_thread.start()
        
        # Motor controller thread (Thread 3) already started
        
        print("✓ All threads started")
        return True
    
    def _camera_loop(self) -> None:
        """
        Thread 1: Camera and depth estimation.
        Runs as fast as possible (30-60 Hz depending on hardware).
        """
        print("[CameraThread] Started")
        
        while self.running:
            loop_start = time.time()
            
            try:
                # Capture frame
                ret, frame = self.video_source.read()
                if not ret:
                    print("[CameraThread] Failed to capture frame")
                    time.sleep(0.1)
                    continue
                
                # Store frame for visualization
                with self.frame_lock:
                    self.latest_frame = frame.copy()
                
                # Estimate depth
                depth_map, _, disp_map = self.depth_processor.estimate_depth(frame)
                
                # Generate point cloud
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                points, colors = self.depth_processor.depth_to_pointcloud(
                    depth_map,
                    frame_rgb,
                    self.config['video']
                )
                
                # Process point cloud (filtering, downsampling)
                points_filtered, colors_filtered = self.pc_processor.process(points, colors)
                
                # Update shared data
                with self.points_lock:
                    self.latest_points = points_filtered
                    self.latest_colors = colors_filtered
                
                # Track FPS
                elapsed = time.time() - loop_start
                fps = 1.0 / elapsed if elapsed > 0 else 0
                self.camera_fps.append(fps)
                self.frame_count += 1
                
            except Exception as e:
                print(f"[CameraThread] Error: {e}")
                time.sleep(0.1)
        
        print("[CameraThread] Stopped")
    
    def _navigation_loop(self) -> None:
        """
        Thread 2: Navigation processing.
        Runs at fixed 30 Hz rate.
        """
        print(f"[NavigationThread] Started at {self.target_nav_rate} Hz")
        
        while self.running:
            loop_start = time.time()
            
            try:
                # Get latest point cloud
                with self.points_lock:
                    points = self.latest_points.copy() if self.latest_points is not None else None
                    colors = self.latest_colors.copy() if self.latest_colors is not None else None
                
                if points is None or len(points) == 0:
                    # No data yet or empty cloud - send stop command
                    with self.command_lock:
                        self.latest_nav_command = {
                            'linear_velocity': 0.0,
                            'angular_velocity': 0.0,
                            'emergency_stop': True
                        }
                    time.sleep(self.nav_period)
                    continue
                
                # Update occupancy grid
                self.occupancy_grid.update(points)
                
                # Compute navigation command using VFH+
                nav_command = self.navigator.compute_navigation_command(points)
                
                # Store command
                with self.command_lock:
                    self.latest_nav_command = nav_command
                
                # Send to motor controller
                if not self.emergency_stop_flag and not nav_command.get('emergency_stop', False):
                    self.motor_controller.send_velocity_command(
                        nav_command['linear_velocity'],
                        nav_command['angular_velocity']
                    )
                else:
                    self.motor_controller.emergency_stop()
                
                # Track FPS
                elapsed = time.time() - loop_start
                fps = 1.0 / elapsed if elapsed > 0 else 0
                self.navigation_fps.append(fps)
                self.nav_count += 1
                
                # Sleep to maintain target rate
                sleep_time = self.nav_period - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                print(f"[NavigationThread] Error: {e}")
                self.motor_controller.emergency_stop()
                time.sleep(self.nav_period)
        
        print("[NavigationThread] Stopped")
    
    def run_main_loop(self) -> None:
        """
        Main loop for visualization and user input.
        Runs in main thread.
        """
        cv2.namedWindow('Navigation Display', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Navigation Display', 1280, 720)
        
        print("\n" + "="*70)
        print(" Controls:")
        print("   Q / ESC     - Emergency stop and quit")
        print("   P           - Pause/Resume")
        print("   R           - Reset navigation")
        print("   Arrow Keys  - Set goal direction")
        print("   S           - Show statistics")
        print("="*70 + "\n")
        
        paused = False
        show_stats = False
        
        try:
            while self.running:
                # Get latest frame and navigation data
                with self.frame_lock:
                    frame = self.latest_frame.copy() if self.latest_frame is not None else None
                
                with self.command_lock:
                    nav_cmd = self.latest_nav_command.copy() if self.latest_nav_command is not None else None
                
                if frame is not None and nav_cmd is not None:
                    # Create visualization
                    display = self._create_visualization(frame, nav_cmd, show_stats)
                    cv2.imshow('Navigation Display', display)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # Q or ESC
                    print("\n[MAIN] Emergency stop initiated by user!")
                    self.emergency_stop_flag = True
                    self.motor_controller.emergency_stop()
                    time.sleep(0.5)
                    break
                    
                elif key == ord('p'):
                    paused = not paused
                    if paused:
                        self.motor_controller.emergency_stop()
                    print(f"[MAIN] {'Paused' if paused else 'Resumed'}")
                    
                elif key == ord('r'):
                    print("[MAIN] Navigation reset")
                    self.navigator.reset()
                    self.occupancy_grid.clear()
                    self.pc_processor.reset_temporal_buffer()
                    
                elif key == ord('s'):
                    show_stats = not show_stats
                    
                elif key == 82:  # Up arrow
                    self.navigator.set_goal_direction(0.0)
                    print("[MAIN] Goal: Forward")
                    
                elif key == 81:  # Left arrow
                    self.navigator.set_goal_direction(np.radians(45))
                    print("[MAIN] Goal: Left 45°")
                    
                elif key == 83:  # Right arrow
                    self.navigator.set_goal_direction(np.radians(-45))
                    print("[MAIN] Goal: Right 45°")
                
                time.sleep(0.01)  # Small delay
                
        except KeyboardInterrupt:
            print("\n[MAIN] Keyboard interrupt")
            self.emergency_stop_flag = True
            self.motor_controller.emergency_stop()
        
        finally:
            self.stop()
    
    def _create_visualization(self, frame: np.ndarray, nav_cmd: Dict, show_stats: bool) -> np.ndarray:
        """Create comprehensive visualization display."""
        h, w = frame.shape[:2]
        
        # Get occupancy grid 2D projection
        occupancy_2d = self.occupancy_grid.get_occupancy_2d(0.0, 2.0)
        
        # Visualize occupancy grid
        occ_vis = (occupancy_2d * 255).astype(np.uint8)
        occ_vis = cv2.applyColorMap(occ_vis, cv2.COLORMAP_JET)
        occ_vis = cv2.resize(occ_vis, (w, h))
        
        # Draw navigation arrow on occupancy grid
        center_x, center_y = w // 2, h // 2
        arrow_len = 80
        safe_dir = nav_cmd.get('safe_direction', 0.0)
        end_x = int(center_x + arrow_len * np.sin(safe_dir))
        end_y = int(center_y - arrow_len * np.cos(safe_dir))
        
        arrow_color = (0, 255, 0) if not nav_cmd.get('emergency_stop', False) else (0, 0, 255)
        cv2.arrowedLine(occ_vis, (center_x, center_y), (end_x, end_y), arrow_color, 3)
        
        # Combine frame and occupancy grid
        display = np.hstack((frame, occ_vis))
        
        # Overlay information
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 30
        
        # Camera FPS
        cam_fps = np.mean(self.camera_fps) if self.camera_fps else 0
        nav_fps = np.mean(self.navigation_fps) if self.navigation_fps else 0
        
        info_lines = [
            f"Camera: {cam_fps:.1f} FPS | Nav: {nav_fps:.1f} FPS",
            f"Vel: {nav_cmd.get('linear_velocity', 0):.2f} m/s | Yaw: {np.degrees(nav_cmd.get('angular_velocity', 0)):.1f} deg/s",
            f"Obstacle: {nav_cmd.get('obstacle_distance', 0):.2f} m",
            f"Mode: {'LIVE' if self.enable_motors else 'SIM'} | Status: {'STOP' if nav_cmd.get('emergency_stop', False) else 'GO'}"
        ]
        
        for i, line in enumerate(info_lines):
            y = y_offset + i * 30
            cv2.putText(display, line, (10, y), font, 0.6, (0, 0, 0), 3)
            cv2.putText(display, line, (10, y), font, 0.6, (0, 255, 0), 1)
        
        # Extended statistics
        if show_stats:
            stats_y = h - 200
            pc_stats = self.pc_processor.get_statistics()
            nav_stats = {
                'avg_comp_time_ms': self.navigator.get_average_computation_time()
            }
            motor_stats = self.motor_controller.get_statistics()
            
            stats_lines = [
                f"PC Processing: {pc_stats['avg_processing_time_ms']:.1f} ms",
                f"Navigation: {nav_stats['avg_comp_time_ms']:.1f} ms",
                f"Motor Cmds: {motor_stats['commands_sent']}",
                f"Errors: {motor_stats['transmission_errors']}"
            ]
            
            for i, line in enumerate(stats_lines):
                y = stats_y + i * 25
                cv2.putText(display, line, (10, y), font, 0.5, (255, 255, 0), 1)
        
        return display
    
    def stop(self) -> None:
        """Stop all threads and clean up."""
        print("\n[MAIN] Shutting down system...")
        
        self.running = False
        
        # Wait for threads
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=1.0)
        
        if self.navigation_thread and self.navigation_thread.is_alive():
            self.navigation_thread.join(timeout=1.0)
        
        # Stop motor controller
        self.motor_controller.stop()
        
        # Release video
        if self.video_source:
            self.video_source.release()
        
        cv2.destroyAllWindows()
        
        # Print final statistics
        print("\n" + "="*70)
        print(" Final Statistics:")
        print(f"   Frames processed: {self.frame_count}")
        print(f"   Navigation cycles: {self.nav_count}")
        if self.camera_fps:
            print(f"   Average camera FPS: {np.mean(self.camera_fps):.2f}")
        if self.navigation_fps:
            print(f"   Average navigation FPS: {np.mean(self.navigation_fps):.2f}")
        print("="*70)
        print("\nSystem shutdown complete.\n")


def main():
    parser = argparse.ArgumentParser(
        description='Autonomous Drone Navigation System - Jetson Optimized',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--weight_path', type=str, required=True,
                       help='Path to RTMonoDepth weights')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to configuration file')
    parser.add_argument('--source', type=str, default='0',
                       help='Video source (webcam index, RTSP URL, or file)')
    parser.add_argument('--enable_motors', action='store_true',
                       help='Enable actual motor control (default: simulation)')
    parser.add_argument('--no_cuda', action='store_true',
                       help='Disable CUDA acceleration')
    
    args = parser.parse_args()
    
    # Create system
    system = AutonomousDroneSystem(
        weight_path=args.weight_path,
        config_path=args.config,
        source=args.source,
        use_cuda=not args.no_cuda,
        enable_motors=args.enable_motors
    )
    
    # Start threads
    if not system.start():
        print("Failed to start system")
        return 1
    
    # Run main loop
    system.run_main_loop()
    
    return 0


if __name__ == '__main__':
    exit(main())
