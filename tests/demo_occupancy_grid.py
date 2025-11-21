"""
Quick Occupancy Grid Visualization Script
Simple script to visualize occupancy grid with sample point clouds
"""

import numpy as np
import sys
import os
import cv2

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.navigation.occupancy_grid import OccupancyGrid3D


def visualize_grid_simple(grid: OccupancyGrid3D, title="Occupancy Grid"):
    """
    Simple visualization of occupancy grid
    
    Args:
        grid: OccupancyGrid3D instance
        title: Window title
    """
    # Get 2D projection
    grid_2d = grid.get_occupancy_2d(z_min=0.0, z_max=3.0)
    
    # Normalize to 0-255
    grid_vis = (grid_2d * 255).astype(np.uint8)
    
    # Apply colormap (JET: blue=free, red=occupied)
    grid_colored = cv2.applyColorMap(grid_vis, cv2.COLORMAP_JET)
    
    # Resize for better visibility
    grid_colored = cv2.resize(grid_colored, (600, 600), interpolation=cv2.INTER_NEAREST)
    
    # Rotate so forward is up
    grid_colored = cv2.rotate(grid_colored, cv2.ROTATE_90_COUNTERCLOCKWISE)
    grid_colored = cv2.flip(grid_colored, 1)
    
    # Draw drone position at center
    h, w = grid_colored.shape[:2]
    cx, cy = w // 2, h // 2
    
    # Draw crosshair
    cv2.line(grid_colored, (cx - 15, cy), (cx + 15, cy), (0, 255, 0), 2)
    cv2.line(grid_colored, (cx, cy - 15), (cx, cy + 15), (0, 255, 0), 2)
    cv2.circle(grid_colored, (cx, cy), 8, (0, 255, 0), 2)
    
    # Add direction indicator
    cv2.arrowedLine(grid_colored, (cx, cy), (cx, cy - 40), (0, 255, 0), 2, tipLength=0.3)
    cv2.putText(grid_colored, 'FORWARD', (cx - 35, cy - 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Add grid info
    stats = grid.get_statistics()
    info_text = [
        f"Occupied: {stats['occupied_cells']} cells",
        f"Occupancy: {stats['occupancy_ratio']*100:.2f}%",
        f"Resolution: {grid.resolution}m",
        f"Size: {grid.size_x}x{grid.size_y}m"
    ]
    
    y_offset = 25
    for text in info_text:
        cv2.putText(grid_colored, text, (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
    
    # Add color legend
    legend_h = 20
    legend_w = 200
    legend_x = w - legend_w - 10
    legend_y = h - legend_h - 10
    
    # Create gradient
    gradient = np.linspace(0, 255, legend_w, dtype=np.uint8)
    gradient = np.tile(gradient, (legend_h, 1))
    gradient_colored = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
    
    # Place legend
    grid_colored[legend_y:legend_y+legend_h, legend_x:legend_x+legend_w] = gradient_colored
    
    cv2.putText(grid_colored, 'Free', (legend_x, legend_y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(grid_colored, 'Occupied', (legend_x + legend_w - 50, legend_y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Show window
    cv2.imshow(title, grid_colored)
    cv2.waitKey(1)
    
    return grid_colored


def main():
    """Demo with different obstacle scenarios"""
    print("\n" + "="*70)
    print("OCCUPANCY GRID VISUALIZATION DEMO")
    print("="*70)
    print("\nThis demo shows how the occupancy grid represents obstacles.")
    print("\nPress any key to advance through scenarios...")
    print("Press Q to quit\n")
    
    # Create grid
    config = {
        'grid_resolution': 0.1,
        'grid_size_x': 10.0,
        'grid_size_y': 10.0,
        'grid_size_z': 3.0,
        'occupancy_threshold': 0.5,
        'decay_rate': 0.9,
        'compute_distance_transform': True
    }
    
    grid = OccupancyGrid3D(config)
    
    # Scenario 1: Wall in front
    print("\n[Scenario 1] Wall directly ahead at 3m")
    grid.clear()
    
    wall_points = []
    for _ in range(500):
        x = 3.0 + np.random.normal(0, 0.1)
        y = np.random.uniform(-2, 2)
        z = np.random.uniform(0.5, 2.0)
        wall_points.append([x, y, z])
    
    grid.update(np.array(wall_points))
    visualize_grid_simple(grid, "Scenario 1: Wall Ahead")
    
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
        return
    
    # Scenario 2: Obstacle on left
    print("\n[Scenario 2] Obstacle on left side")
    grid.clear()
    
    left_obstacle = []
    for _ in range(300):
        x = 2.0 + np.random.normal(0, 0.2)
        y = -2.0 + np.random.normal(0, 0.3)
        z = np.random.uniform(0.5, 1.5)
        left_obstacle.append([x, y, z])
    
    grid.update(np.array(left_obstacle))
    visualize_grid_simple(grid, "Scenario 2: Left Obstacle")
    
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
        return
    
    # Scenario 3: Narrow corridor
    print("\n[Scenario 3] Narrow corridor")
    grid.clear()
    
    corridor = []
    # Left wall
    for _ in range(300):
        x = np.random.uniform(1, 4)
        y = -1.5 + np.random.normal(0, 0.1)
        z = np.random.uniform(0.5, 2.0)
        corridor.append([x, y, z])
    
    # Right wall
    for _ in range(300):
        x = np.random.uniform(1, 4)
        y = 1.5 + np.random.normal(0, 0.1)
        z = np.random.uniform(0.5, 2.0)
        corridor.append([x, y, z])
    
    grid.update(np.array(corridor))
    visualize_grid_simple(grid, "Scenario 3: Narrow Corridor")
    
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
        return
    
    # Scenario 4: Complex environment
    print("\n[Scenario 4] Complex environment with multiple obstacles")
    grid.clear()
    
    complex_env = []
    
    # Wall ahead
    for _ in range(200):
        x = 3.5 + np.random.normal(0, 0.1)
        y = np.random.uniform(-2, 0)
        z = np.random.uniform(0.5, 2.0)
        complex_env.append([x, y, z])
    
    # Box on right
    for _ in range(150):
        x = 2.0 + np.random.normal(0, 0.2)
        y = 1.5 + np.random.normal(0, 0.2)
        z = np.random.uniform(0.5, 1.5)
        complex_env.append([x, y, z])
    
    # Pole on left
    for _ in range(100):
        x = 1.5 + np.random.normal(0, 0.05)
        y = -1.0 + np.random.normal(0, 0.05)
        z = np.linspace(0, 2.0, 100)[_]
        complex_env.append([x, y, z])
    
    # Scattered points
    for _ in range(100):
        x = np.random.uniform(1, 4)
        y = np.random.uniform(-2, 2)
        z = np.random.uniform(0, 2)
        complex_env.append([x, y, z])
    
    grid.update(np.array(complex_env))
    visualize_grid_simple(grid, "Scenario 4: Complex Environment")
    
    print("\n[Scenario 4] Statistics:")
    stats = grid.get_statistics()
    print(f"  - Occupied cells: {stats['occupied_cells']}")
    print(f"  - Occupancy ratio: {stats['occupancy_ratio']*100:.2f}%")
    print(f"  - Update time: {stats['avg_update_time_ms']:.2f}ms")
    
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
        return
    
    # Scenario 5: Decay demonstration
    print("\n[Scenario 5] Demonstrating temporal decay")
    print("  Watch as obstacles fade over time with no new observations...")
    
    for i in range(20):
        grid.update(np.empty((0, 3)))  # Empty update = decay
        visualize_grid_simple(grid, f"Scenario 5: Decay (frame {i+1}/20)")
        
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            break
    
    print("\n" + "="*70)
    print("Demo complete!")
    print("="*70)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
