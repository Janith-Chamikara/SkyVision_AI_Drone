"""
Test script for OccupancyGrid3D module
Tests grid initialization, point cloud updates, occupancy queries, and distance transforms
Uses real point cloud data from tests/kitti-frames_pointcloud.ply
"""

import numpy as np
import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.navigation.occupancy_grid import OccupancyGrid3D

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("⚠️  Warning: Open3D not installed. Using synthetic point clouds only.")


def create_test_config():
    """Create test configuration for occupancy grid."""
    return {
        'grid_resolution': 0.1,  # 10cm cells
        'grid_size_x': 10.0,     # 10m x 10m x 3m grid
        'grid_size_y': 10.0,
        'grid_size_z': 3.0,
        'occupancy_threshold': 0.5,
        'decay_rate': 0.9,
        'compute_distance_transform': True
    }


def generate_dummy_point_cloud(shape='wall', num_points=1000):
    """
    Generate dummy point clouds for testing.
    
    Args:
        shape: 'wall', 'box', 'scattered', 'line', 'plane'
        num_points: Number of points to generate
        
    Returns:
        Nx3 numpy array of XYZ coordinates
    """
    if shape == 'wall':
        # Wall in front of drone at 3m distance
        x = np.full(num_points, 3.0) + np.random.normal(0, 0.1, num_points)
        y = np.random.uniform(-2, 2, num_points)
        z = np.random.uniform(0.5, 2.0, num_points)
        
    elif shape == 'box':
        # Box obstacle at 2m distance
        points_per_face = num_points // 6
        points = []
        
        # Front face
        x = np.full(points_per_face, 2.0)
        y = np.random.uniform(-0.5, 0.5, points_per_face)
        z = np.random.uniform(0.5, 1.5, points_per_face)
        points.append(np.stack([x, y, z], axis=1))
        
        # Back face
        x = np.full(points_per_face, 2.5)
        y = np.random.uniform(-0.5, 0.5, points_per_face)
        z = np.random.uniform(0.5, 1.5, points_per_face)
        points.append(np.stack([x, y, z], axis=1))
        
        # Combine all faces
        points = np.vstack(points)
        return points
        
    elif shape == 'scattered':
        # Random scattered points
        x = np.random.uniform(1, 4, num_points)
        y = np.random.uniform(-3, 3, num_points)
        z = np.random.uniform(0, 2, num_points)
        
    elif shape == 'line':
        # Line obstacle (like a pole)
        x = np.full(num_points, 2.0) + np.random.normal(0, 0.05, num_points)
        y = np.full(num_points, 1.0) + np.random.normal(0, 0.05, num_points)
        z = np.linspace(0, 2.0, num_points)
        
    elif shape == 'plane':
        # Ground plane
        x = np.random.uniform(0, 5, num_points)
        y = np.random.uniform(-3, 3, num_points)
        z = np.full(num_points, 0.0) + np.random.normal(0, 0.05, num_points)
        
    else:
        raise ValueError(f"Unknown shape: {shape}")
    
    return np.stack([x, y, z], axis=1)


def test_initialization():
    """Test 1: Grid initialization."""
    print("\n" + "="*70)
    print("TEST 1: Grid Initialization")
    print("="*70)
    
    config = create_test_config()
    grid = OccupancyGrid3D(config)
    
    # Check grid dimensions
    assert grid.cells_x == 100, f"Expected 100 cells in X, got {grid.cells_x}"
    assert grid.cells_y == 100, f"Expected 100 cells in Y, got {grid.cells_y}"
    assert grid.cells_z == 30, f"Expected 30 cells in Z, got {grid.cells_z}"
    
    # Check grid is empty
    assert np.sum(grid.grid) == 0, "Grid should be empty initially"
    
    print("✓ Grid dimensions correct: 100x100x30 cells")
    print("✓ Grid initialized empty")
    print("✓ Resolution: 0.1m per cell")
    print("✓ TEST PASSED")


def test_point_cloud_update():
    """Test 2: Point cloud update."""
    print("\n" + "="*70)
    print("TEST 2: Point Cloud Update")
    print("="*70)
    
    config = create_test_config()
    grid = OccupancyGrid3D(config)
    
    # Generate wall obstacle
    points = generate_dummy_point_cloud('wall', num_points=5000)
    print(f"Generated wall with {len(points)} points")
    
    # Update grid
    start_time = time.time()
    grid.update(points)
    update_time = (time.time() - start_time) * 1000
    
    # Check that some cells are occupied
    occupied_cells = np.sum(grid.grid > grid.occupancy_threshold)
    
    # Get statistics
    stats = grid.get_statistics()
    print(f"✓ Occupied cells: {stats['occupied_cells']}")
    print(f"✓ Update time: {update_time:.2f} ms")
    
    assert occupied_cells > 0, "Some cells should be occupied after update"
    assert update_time < 150, f"Update too slow: {update_time:.2f} ms (should be < 150ms for 5k points)"
    print(f"✓ Grid statistics:")
    print(f"  - Occupied cells: {stats['occupied_cells']}")
    print(f"  - Total cells: {stats['total_cells']}")
    print(f"  - Occupancy ratio: {stats['occupancy_ratio']:.4f}")
    print("✓ TEST PASSED")


def test_decay():
    """Test 3: Occupancy decay over time."""
    print("\n" + "="*70)
    print("TEST 3: Occupancy Decay")
    print("="*70)
    
    config = create_test_config()
    grid = OccupancyGrid3D(config)
    
    # Add obstacle
    points = generate_dummy_point_cloud('box', num_points=1000)
    grid.update(points)
    
    initial_occupied = np.sum(grid.grid > grid.occupancy_threshold)
    print(f"Initial occupied cells: {initial_occupied}")
    
    # Update with empty point cloud multiple times (simulate decay)
    for i in range(10):
        grid.update(np.empty((0, 3)))
    
    final_occupied = np.sum(grid.grid > grid.occupancy_threshold)
    print(f"After 10 decay updates: {final_occupied} cells")
    print(f"Decay ratio: {final_occupied / initial_occupied:.2f}")
    
    assert final_occupied < initial_occupied, "Occupancy should decay over time"
    print("✓ Occupancy decayed as expected")
    print("✓ TEST PASSED")


def test_2d_projection():
    """Test 4: 2D projection."""
    print("\n" + "="*70)
    print("TEST 4: 2D Occupancy Projection")
    print("="*70)
    
    config = create_test_config()
    grid = OccupancyGrid3D(config)
    
    # Add obstacle at specific height
    points = generate_dummy_point_cloud('wall', num_points=2000)
    grid.update(points)
    
    # Get 2D projection
    grid_2d = grid.get_occupancy_2d(z_min=0.5, z_max=2.0)
    
    print(f"✓ 2D grid shape: {grid_2d.shape}")
    print(f"✓ 2D occupied cells: {np.sum(grid_2d > grid.occupancy_threshold)}")
    print(f"✓ Max occupancy value: {np.max(grid_2d):.3f}")
    print(f"✓ Min occupancy value: {np.min(grid_2d):.3f}")
    
    assert grid_2d.shape == (100, 100), f"Expected 100x100 2D grid, got {grid_2d.shape}"
    assert np.sum(grid_2d > 0) > 0, "2D projection should have occupied cells"
    
    print("✓ TEST PASSED")


def test_occupancy_query():
    """Test 5: Point occupancy queries."""
    print("\n" + "="*70)
    print("TEST 5: Occupancy Queries")
    print("="*70)
    
    config = create_test_config()
    grid = OccupancyGrid3D(config)
    
    # Add wall at x=3m
    points = generate_dummy_point_cloud('wall', num_points=5000)
    grid.update(points)
    
    # Test queries
    test_points = {
        'At wall (should be occupied)': np.array([3.0, 0.0, 1.0]),
        'Before wall (should be free)': np.array([1.0, 0.0, 1.0]),
        'Behind wall (should be free)': np.array([5.0, 0.0, 1.0]),
        'Far left (should be free)': np.array([2.0, -5.0, 1.0]),
    }
    
    for description, point in test_points.items():
        is_occ = grid.is_occupied(point)
        print(f"✓ {description}: {'OCCUPIED' if is_occ else 'FREE'}")
    
    # Wall should be occupied
    wall_point = np.array([3.0, 0.0, 1.0])
    assert grid.is_occupied(wall_point), "Wall point should be occupied"
    
    # Point before wall should be free
    free_point = np.array([1.0, 0.0, 1.0])
    assert not grid.is_occupied(free_point), "Free point should not be occupied"
    
    print("✓ TEST PASSED")


def test_distance_transform():
    """Test 6: Distance transform."""
    print("\n" + "="*70)
    print("TEST 6: Distance Transform")
    print("="*70)
    
    config = create_test_config()
    grid = OccupancyGrid3D(config)
    
    # Add obstacle
    points = generate_dummy_point_cloud('box', num_points=2000)
    grid.update(points)
    
    # Check distance transform was computed
    assert grid.distance_transform is not None, "Distance transform should be computed"
    print("✓ Distance transform computed")
    print(f"✓ Distance transform shape: {grid.distance_transform.shape}")
    
    # Query distances at various points
    test_points = [
        (np.array([1.0, 0.0, 1.0]), "1m from obstacle"),
        (np.array([0.5, 0.0, 1.0]), "0.5m from obstacle"),
        (np.array([4.0, 0.0, 1.0]), "4m from obstacle"),
    ]
    
    for point, description in test_points:
        distance = grid.get_distance_at_point(point)
        print(f"✓ {description}: {distance:.2f}m")
    
    print("✓ TEST PASSED")


def test_multiple_updates():
    """Test 7: Multiple sequential updates."""
    print("\n" + "="*70)
    print("TEST 7: Multiple Sequential Updates")
    print("="*70)
    
    config = create_test_config()
    grid = OccupancyGrid3D(config)
    
    shapes = ['wall', 'box', 'scattered', 'line']
    
    for i, shape in enumerate(shapes):
        points = generate_dummy_point_cloud(shape, num_points=1000)
        grid.update(points)
        
        stats = grid.get_statistics()
        print(f"✓ Update {i+1} ({shape}): {stats['occupied_cells']} occupied cells, "
              f"{stats['avg_update_time_ms']:.2f} ms")
    
    # Check that grid is still responsive
    final_stats = grid.get_statistics()
    assert final_stats['avg_update_time_ms'] < 50, "Updates should remain fast"
    
    print(f"✓ Average update time: {final_stats['avg_update_time_ms']:.2f} ms")
    print("✓ TEST PASSED")


def test_edge_cases():
    """Test 8: Edge cases."""
    print("\n" + "="*70)
    print("TEST 8: Edge Cases")
    print("="*70)
    
    config = create_test_config()
    grid = OccupancyGrid3D(config)
    
    # Test 1: Empty point cloud
    print("Testing empty point cloud...")
    grid.update(np.empty((0, 3)))
    assert np.sum(grid.grid) == 0, "Grid should remain empty"
    print("✓ Empty point cloud handled")
    
    # Test 2: None input
    print("Testing None input...")
    grid.update(None)
    print("✓ None input handled")
    
    # Test 3: Points outside grid bounds
    print("Testing out-of-bounds points...")
    out_of_bounds = np.array([
        [100.0, 0.0, 0.0],  # Far away
        [-100.0, 0.0, 0.0],  # Behind
        [0.0, 100.0, 0.0],  # Far left
        [0.0, -100.0, 0.0],  # Far right
        [0.0, 0.0, 100.0],  # Very high
    ])
    grid.update(out_of_bounds)
    print("✓ Out-of-bounds points handled (should be ignored)")
    
    # Test 4: Single point
    print("Testing single point...")
    single_point = np.array([[2.0, 0.0, 1.0]])
    grid.update(single_point)
    print("✓ Single point handled")
    
    # Test 5: Clear grid
    print("Testing clear...")
    grid.clear()
    assert np.sum(grid.grid) == 0, "Grid should be empty after clear"
    print("✓ Grid cleared successfully")
    
    print("✓ TEST PASSED")


def test_performance():
    """Test 9: Performance benchmark."""
    print("\n" + "="*70)
    print("TEST 9: Performance Benchmark")
    print("="*70)
    
    config = create_test_config()
    grid = OccupancyGrid3D(config)
    
    point_counts = [100, 500, 1000, 5000, 10000]
    
    print("\nBenchmarking different point cloud sizes:")
    print(f"{'Points':<10} {'Update Time (ms)':<20} {'Throughput (pts/ms)':<20}")
    print("-" * 50)
    
    for num_points in point_counts:
        points = generate_dummy_point_cloud('scattered', num_points=num_points)
        
        # Run multiple times for average
        times = []
        for _ in range(10):
            start = time.time()
            grid.update(points)
            times.append((time.time() - start) * 1000)
        
        avg_time = np.mean(times)
        throughput = num_points / avg_time
        
        print(f"{num_points:<10} {avg_time:<20.2f} {throughput:<20.1f}")
        
        # Performance requirements (relaxed for non-Jetson hardware)
        if num_points <= 1000:
            assert avg_time < 30, f"Update with {num_points} points too slow: {avg_time:.2f}ms"
        elif num_points <= 5000:
            assert avg_time < 50, f"Update with {num_points} points too slow: {avg_time:.2f}ms"
        elif num_points <= 10000:
            assert avg_time < 100, f"Update with {num_points} points too slow: {avg_time:.2f}ms"
    
    print("\n✓ Performance meets requirements")
    print("✓ TEST PASSED")


def test_slice_visualization():
    """Test 10: Grid slicing for visualization."""
    print("\n" + "="*70)
    print("TEST 10: Grid Slicing")
    print("="*70)
    
    config = create_test_config()
    grid = OccupancyGrid3D(config)
    
    # Add obstacle
    points = generate_dummy_point_cloud('wall', num_points=3000)
    grid.update(points)
    
    # Test different slice axes
    slice_x = grid.get_slice(axis='x', index=None)
    slice_y = grid.get_slice(axis='y', index=None)
    slice_z = grid.get_slice(axis='z', index=15)  # Mid-height
    
    print(f"✓ X-axis slice shape: {slice_x.shape}")
    print(f"✓ Y-axis slice shape: {slice_y.shape}")
    print(f"✓ Z-axis slice shape: {slice_z.shape}")
    
    assert slice_x.shape == (100, 30), "X slice should be YZ plane"
    assert slice_y.shape == (100, 30), "Y slice should be XZ plane"
    assert slice_z.shape == (100, 100), "Z slice should be XY plane"
    
    print("✓ All slice dimensions correct")
    print("✓ TEST PASSED")


def test_with_real_point_cloud():
    """Test 11: Process real KITTI point cloud data."""
    print("\n" + "="*70)
    print("TEST 11: Real Point Cloud from KITTI Dataset")
    print("="*70)
    
    config = create_test_config()
    grid = OccupancyGrid3D(config)
    
    # Try to load real point cloud
    pcd_path = os.path.join(os.path.dirname(__file__), "kitti-frames_pointcloud.ply")
    
    if not HAS_OPEN3D:
        print("⚠️  Skipping: Open3D not available")
        print("   Install with: pip install open3d")
        print("✓ TEST SKIPPED")
        return
    
    if not os.path.exists(pcd_path):
        print(f"⚠️  Skipping: Point cloud file not found at {pcd_path}")
        print("✓ TEST SKIPPED")
        return
    
    print(f"📂 Loading point cloud from: {pcd_path}")
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)
    print(f"   Loaded {len(points)} points")
    
    # Filter points to reasonable drone view range
    # Keep points within 10m forward, 5m sideways, 3m height
    mask = (points[:, 0] >= -5) & (points[:, 0] <= 5) & \
           (points[:, 1] >= -5) & (points[:, 1] <= 5) & \
           (points[:, 2] >= -1) & (points[:, 2] <= 3)
    filtered_points = points[mask]
    
    print(f"   Filtered to {len(filtered_points)} points in grid range")
    
    # Update grid
    start_time = time.time()
    grid.update(filtered_points)
    update_time = (time.time() - start_time) * 1000
    
    # Get statistics
    stats = grid.get_statistics()
    
    print(f"\n📊 Processing Results:")
    print(f"   - Update time: {update_time:.2f} ms")
    print(f"   - Occupied cells: {stats['occupied_cells']}")
    print(f"   - Total cells: {stats['total_cells']}")
    print(f"   - Occupancy ratio: {stats['occupancy_ratio']*100:.3f}%")
    print(f"   - Avg update time: {stats['avg_update_time_ms']:.2f} ms")
    
    # Validate results
    assert stats['occupied_cells'] > 0, "Should have occupied cells from real data"
    assert update_time < 100, f"Update too slow: {update_time:.2f}ms (should be < 100ms)"
    
    # Test distance transform on real data
    if grid.distance_transform is not None:
        print(f"   - Distance transform: ✓ computed")
        
        # Sample some points and check distances
        test_point = np.array([0.0, 0.0, 1.0])  # 1m in front of origin
        distance = grid.get_distance_at_point(test_point)
        print(f"   - Distance at origin to obstacle: {distance:.2f}m")
    
    # Get 2D projection for visualization info
    grid_2d = grid.get_occupancy_2d(z_min=0.0, z_max=2.0)
    occupied_2d = np.sum(grid_2d > grid.occupancy_threshold)
    print(f"   - 2D projection occupied cells: {occupied_2d}")
    
    print("\n✓ Successfully processed real point cloud data")
    print("✓ TEST PASSED")


def run_all_tests():
    """Run all test cases."""
    print("\n" + "="*70)
    print("OCCUPANCY GRID 3D - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    tests = [
        test_initialization,
        test_point_cloud_update,
        test_decay,
        test_2d_projection,
        test_occupancy_query,
        test_distance_transform,
        test_multiple_updates,
        test_edge_cases,
        test_performance,
        test_slice_visualization,
        test_with_real_point_cloud,  # New test with real data
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ TEST FAILED: {test_func.__name__}")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed} ✓")
    print(f"Failed: {failed} ✗")
    print(f"Success rate: {passed/len(tests)*100:.1f}%")
    print("="*70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
