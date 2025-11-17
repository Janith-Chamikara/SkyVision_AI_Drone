"""
Vector Field Histogram Plus (VFH+) Navigation Algorithm
Efficient obstacle avoidance for real-time drone navigation
Computational complexity: < 20ms per cycle
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import time


class VFHPlusNavigator:
    """
    Vector Field Histogram Plus (VFH+) for obstacle avoidance.
    
    Builds a polar histogram showing obstacle density in each direction,
    finds safe valleys, and selects best navigation direction toward goal.
    """
    
    def __init__(self, config: dict):
        """
        Initialize VFH+ navigator.
        
        Args:
            config: Configuration containing:
                - histogram_resolution: Angular resolution in degrees
                - obstacle_threshold: Density threshold for obstacles (0-1)
                - max_velocity: Maximum linear velocity (m/s)
                - max_angular_velocity: Maximum yaw rate (rad/s)
                - safety_distance: Minimum clearance to obstacles (m)
                - planning_horizon: Look-ahead time (seconds)
                - valley_min_width: Minimum angular width for safe valley (degrees)
        """
        self.config = config
        
        # Histogram parameters
        self.histogram_resolution = config.get('histogram_resolution', 5)  # degrees per sector
        self.num_sectors = 360 // self.histogram_resolution
        self.histogram = np.zeros(self.num_sectors, dtype=np.float32)
        
        # Obstacle detection
        self.obstacle_threshold = config.get('obstacle_threshold', 0.3)
        self.safety_distance = config.get('safety_distance', 1.0)
        
        # Velocity limits
        self.max_velocity = config.get('max_velocity', 2.0)
        self.max_angular_velocity = config.get('max_angular_velocity', 1.57)  # 90 deg/s
        
        # Planning
        self.planning_horizon = config.get('planning_horizon', 1.0)
        self.valley_min_width = config.get('valley_min_width', 10)  # degrees
        
        # Goal tracking
        self.goal_direction = 0.0  # radians (0 = forward)
        
        # Velocity smoothing
        self.velocity_history = deque(maxlen=5)
        self.angular_velocity_history = deque(maxlen=5)
        
        # Performance tracking
        self.computation_times = deque(maxlen=30)
        
        # State
        self.current_velocity = 0.0
        self.current_angular_velocity = 0.0
        
        print(f"[VFHPlusNavigator] Initialized:")
        print(f"  - Histogram: {self.num_sectors} sectors @ {self.histogram_resolution}°")
        print(f"  - Velocity limits: {self.max_velocity} m/s, {np.degrees(self.max_angular_velocity)}°/s")
        print(f"  - Safety distance: {self.safety_distance}m")
    
    def set_goal_direction(self, direction_rad: float) -> None:
        """
        Set target direction.
        
        Args:
            direction_rad: Goal direction in radians (0 = forward, + = left, - = right)
        """
        self.goal_direction = direction_rad
    
    def compute_navigation_command(self, points: np.ndarray) -> Dict:
        """
        Compute navigation command from point cloud.
        
        Args:
            points: Nx3 point cloud in drone frame (X=right, Y=down, Z=forward)
            
        Returns:
            Dictionary containing:
                - linear_velocity: Forward velocity (m/s)
                - angular_velocity: Yaw rate (rad/s)
                - safe_direction: Selected direction (radians)
                - obstacle_distance: Minimum distance to obstacles (m)
                - histogram: Polar obstacle histogram
                - valleys: List of safe valleys
                - emergency_stop: Boolean indicating critical obstacle
        """
        start_time = time.time()
        
        # Handle empty point cloud
        if points is None or len(points) == 0:
            return self._emergency_stop()
        
        # Step 1: Build polar histogram
        self.histogram = self._build_polar_histogram(points)
        
        # Step 2: Find safe valleys
        valleys = self._find_safe_valleys(self.histogram)
        
        # Step 3: Check for emergency stop
        min_distance = self._compute_min_obstacle_distance(points)
        if min_distance < self.safety_distance * 0.5:  # Critical distance
            return self._emergency_stop()
        
        if len(valleys) == 0:
            # No safe direction found
            return self._emergency_stop()
        
        # Step 4: Select best direction
        safe_direction = self._select_best_direction(valleys)
        
        # Step 5: Compute velocities
        linear_velocity = self._compute_linear_velocity(safe_direction, self.histogram, min_distance)
        angular_velocity = self._compute_angular_velocity(safe_direction)
        
        # Step 6: Apply smoothing
        linear_velocity = self._smooth_velocity(linear_velocity)
        angular_velocity = self._smooth_angular_velocity(angular_velocity)
        
        # Update state
        self.current_velocity = linear_velocity
        self.current_angular_velocity = angular_velocity
        
        # Track performance
        elapsed = time.time() - start_time
        self.computation_times.append(elapsed)
        
        return {
            'linear_velocity': linear_velocity,
            'angular_velocity': angular_velocity,
            'safe_direction': safe_direction,
            'obstacle_distance': min_distance,
            'histogram': self.histogram.copy(),
            'valleys': valleys,
            'emergency_stop': False,
            'computation_time_ms': elapsed * 1000
        }
    
    def _build_polar_histogram(self, points: np.ndarray) -> np.ndarray:
        """
        Build VFH polar histogram showing obstacle density per angular sector.
        
        For each point, compute bearing angle and update corresponding sector
        with density inversely proportional to distance.
        """
        histogram = np.zeros(self.num_sectors, dtype=np.float32)
        
        for point in points:
            x, y, z = point
            
            # Convert from drone frame to navigation frame
            # Drone frame: X=right, Y=down, Z=forward
            # Navigation: 0° = forward (Z), + = left
            
            # Compute horizontal distance and angle
            distance = np.sqrt(x**2 + z**2)
            
            # Skip points too close (likely noise) or beyond planning horizon
            max_horizon_dist = self.max_velocity * self.planning_horizon
            if distance < 0.1 or distance > max_horizon_dist:
                continue
            
            # Compute bearing angle (0 = forward/Z, positive = left/-X)
            angle_rad = np.arctan2(-x, z)  # Negative X for left
            angle_deg = np.degrees(angle_rad)
            
            # Map to histogram sector [0, 360)
            sector = int((angle_deg + 180) / self.histogram_resolution) % self.num_sectors
            
            # Weight by inverse distance and obstacle height
            # Closer obstacles and obstacles at drone height have higher weight
            distance_weight = 1.0 / max(distance, 0.1)
            
            # Height consideration (Y axis, down is positive)
            height_weight = 1.0
            if y < -0.5 or y > 1.0:  # Outside drone height band
                height_weight = 0.5
            
            weight = distance_weight * height_weight
            histogram[sector] += weight
        
        # Normalize histogram to [0, 1]
        if histogram.max() > 0:
            histogram = histogram / histogram.max()
        
        # Apply Gaussian smoothing to histogram for robustness
        histogram = self._smooth_histogram(histogram)
        
        return histogram
    
    def _smooth_histogram(self, histogram: np.ndarray, sigma: int = 2) -> np.ndarray:
        """
        Apply circular Gaussian smoothing to histogram.
        
        Reduces noise and creates smoother valleys.
        """
        kernel_size = sigma * 3
        kernel = np.exp(-0.5 * (np.arange(-kernel_size, kernel_size + 1) / sigma) ** 2)
        kernel = kernel / kernel.sum()
        
        # Circular convolution (wrap around)
        padded = np.concatenate([histogram[-kernel_size:], histogram, histogram[:kernel_size]])
        smoothed = np.convolve(padded, kernel, mode='valid')
        
        return smoothed
    
    def _find_safe_valleys(self, histogram: np.ndarray) -> List[Dict]:
        """
        Find continuous angular sectors with low obstacle density (valleys).
        
        Returns list of valleys with start, end, center, and width.
        """
        valleys = []
        in_valley = False
        valley_start = 0
        
        min_width_sectors = self.valley_min_width // self.histogram_resolution
        
        for i in range(self.num_sectors):
            if histogram[i] < self.obstacle_threshold:
                if not in_valley:
                    in_valley = True
                    valley_start = i
            else:
                if in_valley:
                    # End of valley
                    valley_end = i - 1
                    valley_width = valley_end - valley_start + 1
                    
                    # Only keep valleys wide enough
                    if valley_width >= min_width_sectors:
                        valley_center = (valley_start + valley_end) / 2
                        valleys.append({
                            'start': valley_start,
                            'end': valley_end,
                            'center': valley_center,
                            'width': valley_width * self.histogram_resolution  # degrees
                        })
                    in_valley = False
        
        # Handle wrap-around valley
        if in_valley:
            valley_end = self.num_sectors - 1
            valley_width = valley_end - valley_start + 1
            
            if valley_width >= min_width_sectors:
                valley_center = (valley_start + valley_end) / 2
                valleys.append({
                    'start': valley_start,
                    'end': valley_end,
                    'center': valley_center,
                    'width': valley_width * self.histogram_resolution
                })
        
        return valleys
    
    def _select_best_direction(self, valleys: List[Dict]) -> float:
        """
        Select best valley closest to goal direction while maximizing safety.
        
        Scoring considers:
        1. Alignment with goal direction
        2. Valley width (wider is safer)
        3. Current direction (prefer small corrections)
        """
        if not valleys:
            return 0.0
        
        goal_sector = int((np.degrees(self.goal_direction) + 180) / self.histogram_resolution)
        current_sector = int((np.degrees(self.current_angular_velocity) + 180) / self.histogram_resolution)
        
        best_valley = None
        best_score = -float('inf')
        
        for valley in valleys:
            center_sector = valley['center']
            
            # Angular deviation from goal
            goal_deviation = abs(center_sector - goal_sector)
            if goal_deviation > self.num_sectors / 2:
                goal_deviation = self.num_sectors - goal_deviation
            goal_score = 1.0 - (goal_deviation / (self.num_sectors / 2))
            
            # Valley width score (prefer wider valleys)
            width_score = min(valley['width'] / 90.0, 1.0)  # Normalize to 90 degrees
            
            # Continuity score (prefer small changes)
            continuity_deviation = abs(center_sector - current_sector)
            if continuity_deviation > self.num_sectors / 2:
                continuity_deviation = self.num_sectors - continuity_deviation
            continuity_score = 1.0 - (continuity_deviation / (self.num_sectors / 2))
            
            # Combined score (weighted)
            score = (0.5 * goal_score + 
                    0.3 * width_score + 
                    0.2 * continuity_score)
            
            if score > best_score:
                best_score = score
                best_valley = valley
        
        if best_valley:
            # Convert sector back to radians
            angle_deg = best_valley['center'] * self.histogram_resolution - 180
            return np.radians(angle_deg)
        
        return 0.0
    
    def _compute_linear_velocity(self, direction: float, histogram: np.ndarray, 
                                 min_distance: float) -> float:
        """
        Compute forward velocity based on obstacle proximity and direction safety.
        
        Reduces speed when:
        - Obstacles are close
        - Selected direction has high obstacle density
        - Turning sharply
        """
        # Base velocity
        velocity = self.max_velocity
        
        # Reduce speed based on obstacle distance
        if min_distance < self.safety_distance * 2:
            distance_factor = min_distance / (self.safety_distance * 2)
            velocity *= max(distance_factor, 0.2)  # Min 20% speed
        
        # Reduce speed based on obstacle density in selected direction
        direction_sector = int((np.degrees(direction) + 180) / self.histogram_resolution) % self.num_sectors
        direction_density = histogram[direction_sector]
        density_factor = 1.0 - direction_density
        velocity *= max(density_factor, 0.3)  # Min 30% speed
        
        # Reduce speed when turning
        turn_angle = abs(direction)
        if turn_angle > np.radians(30):
            turn_factor = 1.0 - (turn_angle / np.pi)
            velocity *= max(turn_factor, 0.4)  # Min 40% speed when turning
        
        return velocity
    
    def _compute_angular_velocity(self, target_direction: float) -> float:
        """
        Compute yaw rate to reach target direction using proportional control.
        """
        # Direction error (target relative to current heading)
        direction_error = target_direction
        
        # Proportional control
        kp = 1.5  # Proportional gain
        angular_velocity = kp * direction_error
        
        # Clamp to max angular velocity
        angular_velocity = np.clip(angular_velocity, 
                                   -self.max_angular_velocity, 
                                   self.max_angular_velocity)
        
        return angular_velocity
    
    def _compute_min_obstacle_distance(self, points: np.ndarray) -> float:
        """Compute minimum distance to any obstacle point."""
        if len(points) == 0:
            return float('inf')
        
        # Distance in XZ plane (horizontal)
        distances = np.sqrt(points[:, 0]**2 + points[:, 2]**2)
        return np.min(distances)
    
    def _smooth_velocity(self, velocity: float) -> float:
        """Apply temporal smoothing to linear velocity."""
        self.velocity_history.append(velocity)
        return np.mean(self.velocity_history)
    
    def _smooth_angular_velocity(self, angular_velocity: float) -> float:
        """Apply temporal smoothing to angular velocity."""
        self.angular_velocity_history.append(angular_velocity)
        return np.mean(self.angular_velocity_history)
    
    def _emergency_stop(self) -> Dict:
        """Return emergency stop command."""
        return {
            'linear_velocity': 0.0,
            'angular_velocity': 0.0,
            'safe_direction': 0.0,
            'obstacle_distance': 0.0,
            'histogram': self.histogram.copy(),
            'valleys': [],
            'emergency_stop': True,
            'computation_time_ms': 0.0
        }
    
    def get_average_computation_time(self) -> float:
        """Get average computation time in milliseconds."""
        if not self.computation_times:
            return 0.0
        return np.mean(self.computation_times) * 1000
    
    def reset(self) -> None:
        """Reset navigator state."""
        self.velocity_history.clear()
        self.angular_velocity_history.clear()
        self.current_velocity = 0.0
        self.current_angular_velocity = 0.0
        self.histogram.fill(0.0)
