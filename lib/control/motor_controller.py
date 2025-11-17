"""
Motor Control Module
Converts navigation commands to motor PWM signals
Handles serial communication with flight controller
"""

import numpy as np
import serial
import struct
import time
import threading
from typing import Dict, Optional
from collections import deque


class QuadcopterMotorController:
    """
    Motor controller for X-configuration quadcopter.
    
    Converts velocity commands to individual motor PWM values and
    transmits via UART to flight controller.
    """
    
    def __init__(self, config: dict):
        """
        Initialize motor controller.
        
        Args:
            config: Configuration containing:
                - port: Serial port path (e.g., '/dev/ttyTHS1')
                - baudrate: Serial baudrate
                - control_rate: Command transmission rate (Hz)
                - max_tilt_angle: Maximum pitch/roll angle (degrees)
                - hover_throttle: Base throttle for hovering (0-255)
                - min_throttle: Minimum motor speed (0-255)
                - max_throttle: Maximum motor speed (0-255)
        """
        self.config = config
        
        # Serial port configuration
        self.port = config.get('port', '/dev/ttyTHS1')
        self.baudrate = config.get('baudrate', 115200)
        self.serial_conn = None
        
        # Control parameters
        self.control_rate = config.get('control_rate', 50)  # Hz
        self.control_period = 1.0 / self.control_rate
        
        # Quadcopter parameters
        self.max_tilt_angle = np.radians(config.get('max_tilt_angle', 30))
        self.hover_throttle = config.get('hover_throttle', 128)
        self.min_throttle = config.get('min_throttle', 0)
        self.max_throttle = config.get('max_throttle', 255)
        
        # Control gains
        self.velocity_to_pitch_gain = config.get('velocity_to_pitch_gain', 0.1)
        self.angular_velocity_to_yaw_gain = config.get('angular_velocity_to_yaw_gain', 30)
        
        # Threading
        self.running = False
        self.send_thread = None
        self.command_queue = deque(maxlen=1)  # Keep only latest command
        self.lock = threading.Lock()
        
        # Safety
        self.emergency_stop_flag = False
        self.last_command_time = time.time()
        self.command_timeout = 0.5  # seconds
        
        # Current state
        self.current_altitude = 1.0  # meters (estimate)
        self.current_motors = {'motor_1': 0, 'motor_2': 0, 'motor_3': 0, 'motor_4': 0}
        
        # Statistics
        self.commands_sent = 0
        self.transmission_errors = 0
        
        print(f"[QuadcopterMotorController] Initialized:")
        print(f"  - Port: {self.port} @ {self.baudrate} baud")
        print(f"  - Control rate: {self.control_rate} Hz")
        print(f"  - Max tilt: {np.degrees(self.max_tilt_angle)}°")
    
    def connect(self) -> bool:
        """
        Establish serial connection to flight controller.
        
        Returns:
            True if connection successful
        """
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0.1,
                write_timeout=0.1
            )
            print(f"✓ Connected to flight controller on {self.port}")
            return True
        except serial.SerialException as e:
            print(f"✗ Failed to connect to {self.port}: {e}")
            print(f"  Note: For simulation, run without --enable_motors flag")
            return False
        except Exception as e:
            print(f"✗ Unexpected error connecting to {self.port}: {e}")
            return False
    
    def start(self) -> bool:
        """
        Start control loop in background thread.
        
        Returns:
            True if started successfully
        """
        if not self.serial_conn or not self.serial_conn.is_open:
            if not self.connect():
                return False
        
        self.running = True
        self.send_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.send_thread.start()
        print("[MotorController] Control loop started")
        return True
    
    def stop(self) -> None:
        """Stop control loop and close connection."""
        print("[MotorController] Stopping...")
        self.running = False
        
        if self.send_thread and self.send_thread.is_alive():
            self.send_thread.join(timeout=1.0)
        
        if self.serial_conn and self.serial_conn.is_open:
            # Send stop command before closing
            self.emergency_stop()
            time.sleep(0.1)
            self.serial_conn.close()
        
        print("[MotorController] Stopped")
        print(f"  Commands sent: {self.commands_sent}")
        print(f"  Transmission errors: {self.transmission_errors}")
    
    def send_velocity_command(self, linear_velocity: float, angular_velocity: float) -> None:
        """
        Queue velocity command for transmission.
        
        Args:
            linear_velocity: Forward velocity in m/s
            angular_velocity: Yaw rate in rad/s
        """
        # Convert to motor commands
        motors = self.velocity_to_motor_commands(linear_velocity, angular_velocity)
        
        # Queue for transmission
        with self.lock:
            self.command_queue.append(motors)
            self.last_command_time = time.time()
    
    def velocity_to_motor_commands(self, linear_vel: float, angular_vel: float) -> Dict[str, int]:
        """
        Convert velocity commands to individual motor PWM values.
        
        Uses standard quadcopter mixing for X-configuration:
        - Forward velocity → Pitch angle → Differential thrust
        - Yaw rate → Differential motor speeds
        
        Args:
            linear_vel: Forward velocity (m/s)
            angular_vel: Yaw rate (rad/s)
            
        Returns:
            Dictionary with motor_1, motor_2, motor_3, motor_4 PWM values (0-255)
        """
        # Compute pitch angle from desired forward velocity
        # Small angle approximation: pitch ≈ velocity / (g * time_constant)
        pitch_angle = linear_vel * self.velocity_to_pitch_gain
        pitch_angle = np.clip(pitch_angle, -self.max_tilt_angle, self.max_tilt_angle)
        
        # Convert pitch angle to motor correction
        # Positive pitch = nose up = need to speed up front motors
        pitch_correction = int(pitch_angle * 50)  # Scale to PWM range
        
        # Convert yaw rate to motor correction
        # Positive yaw = turn left = speed up motors 1,3, slow down 2,4
        yaw_correction = int(angular_vel * self.angular_velocity_to_yaw_gain)
        
        # Base throttle (hovering)
        base_throttle = self.hover_throttle
        
        # Motor mixing for X-configuration quadcopter
        #     1 (FL)      2 (FR)
        #          \ __ /
        #          /    \
        #     3 (RL)      4 (RR)
        #
        # Motor spin directions:
        # 1: CW, 2: CCW, 3: CCW, 4: CW
        
        motor_1 = base_throttle + yaw_correction + pitch_correction   # Front-left
        motor_2 = base_throttle - yaw_correction + pitch_correction   # Front-right
        motor_3 = base_throttle + yaw_correction - pitch_correction   # Rear-left
        motor_4 = base_throttle - yaw_correction - pitch_correction   # Rear-right
        
        # Clamp to valid PWM range
        motors = {
            'motor_1': int(np.clip(motor_1, self.min_throttle, self.max_throttle)),
            'motor_2': int(np.clip(motor_2, self.min_throttle, self.max_throttle)),
            'motor_3': int(np.clip(motor_3, self.min_throttle, self.max_throttle)),
            'motor_4': int(np.clip(motor_4, self.min_throttle, self.max_throttle))
        }
        
        return motors
    
    def emergency_stop(self) -> None:
        """
        Send immediate stop command to all motors.
        Sets all motor speeds to minimum (typically 0).
        """
        self.emergency_stop_flag = True
        
        stop_motors = {
            'motor_1': self.min_throttle,
            'motor_2': self.min_throttle,
            'motor_3': self.min_throttle,
            'motor_4': self.min_throttle
        }
        
        # Send immediately (bypass queue)
        if self.serial_conn and self.serial_conn.is_open:
            self._transmit_motors(stop_motors)
        
        print("[MotorController] EMERGENCY STOP activated!")
    
    def _control_loop(self) -> None:
        """
        Background thread that sends motor commands at fixed rate.
        """
        print(f"[MotorController] Control loop running at {self.control_rate} Hz")
        
        while self.running:
            loop_start = time.time()
            
            # Check for command timeout
            if time.time() - self.last_command_time > self.command_timeout:
                # No recent commands - send stop
                motors = {
                    'motor_1': self.min_throttle,
                    'motor_2': self.min_throttle,
                    'motor_3': self.min_throttle,
                    'motor_4': self.min_throttle
                }
            else:
                # Get latest command from queue
                with self.lock:
                    if self.command_queue:
                        motors = self.command_queue[-1]
                        self.command_queue.clear()
                    else:
                        motors = self.current_motors
            
            # Check emergency stop flag
            if self.emergency_stop_flag:
                motors = {
                    'motor_1': self.min_throttle,
                    'motor_2': self.min_throttle,
                    'motor_3': self.min_throttle,
                    'motor_4': self.min_throttle
                }
            
            # Transmit to flight controller
            if self.serial_conn and self.serial_conn.is_open:
                success = self._transmit_motors(motors)
                if success:
                    self.current_motors = motors
                    self.commands_sent += 1
            
            # Sleep to maintain control rate
            elapsed = time.time() - loop_start
            sleep_time = self.control_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _transmit_motors(self, motors: Dict[str, int]) -> bool:
        """
        Transmit motor commands over serial port.
        
        Protocol: [START_BYTE, M1, M2, M3, M4, CHECKSUM, END_BYTE]
        
        Args:
            motors: Dictionary with motor PWM values
            
        Returns:
            True if transmission successful
        """
        try:
            # Build packet
            start_byte = 0xAA
            m1 = motors['motor_1'] & 0xFF
            m2 = motors['motor_2'] & 0xFF
            m3 = motors['motor_3'] & 0xFF
            m4 = motors['motor_4'] & 0xFF
            
            # Compute checksum (simple sum)
            checksum = (m1 + m2 + m3 + m4) & 0xFF
            
            end_byte = 0x55
            
            # Pack into bytes
            packet = struct.pack('BBBBBBB', start_byte, m1, m2, m3, m4, checksum, end_byte)
            
            # Send
            self.serial_conn.write(packet)
            self.serial_conn.flush()
            
            return True
            
        except serial.SerialTimeoutException:
            self.transmission_errors += 1
            print(f"[MotorController] Transmission timeout (error {self.transmission_errors})")
            return False
        except Exception as e:
            self.transmission_errors += 1
            print(f"[MotorController] Transmission error: {e}")
            return False
    
    def get_current_motors(self) -> Dict[str, int]:
        """Get current motor speeds."""
        return self.current_motors.copy()
    
    def set_hover_throttle(self, throttle: int) -> None:
        """Update hover throttle value."""
        self.hover_throttle = int(np.clip(throttle, self.min_throttle, self.max_throttle))
    
    def get_statistics(self) -> Dict:
        """Get controller statistics."""
        return {
            'commands_sent': self.commands_sent,
            'transmission_errors': self.transmission_errors,
            'error_rate': self.transmission_errors / max(self.commands_sent, 1),
            'is_connected': self.serial_conn and self.serial_conn.is_open,
            'emergency_stop': self.emergency_stop_flag,
            'current_motors': self.current_motors
        }


class SimulatedMotorController(QuadcopterMotorController):
    """
    Simulated motor controller for testing without hardware.
    
    Logs commands instead of sending to serial port.
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        print("[SimulatedMotorController] Running in SIMULATION mode")
    
    def connect(self) -> bool:
        """Simulate connection."""
        print("✓ Simulated connection established")
        return True
    
    def _transmit_motors(self, motors: Dict[str, int]) -> bool:
        """Log motor commands instead of transmitting."""
        # Optionally print every N commands for debugging
        if self.commands_sent % 50 == 0:
            print(f"[SIM] Motors: M1={motors['motor_1']}, M2={motors['motor_2']}, "
                  f"M3={motors['motor_3']}, M4={motors['motor_4']}")
        return True
