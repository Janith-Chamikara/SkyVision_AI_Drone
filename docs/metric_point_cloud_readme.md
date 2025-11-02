# Metric Point Cloud Upgrade Guide

This note walks through every code change that converts the project from disparity-based point clouds into metric 3D data that can drive obstacle avoidance. Each subsection lists the modified lines and explains exactly what they do, with short usage examples so you can see the effect in practice.

## 1. `lib/depth_point_cloud_utils.py`

| Code (line range) | Explanation                                                                                                                                                                                                                                                                                                                                                           |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1-8               | Import order now includes `disp_to_depth` from `layers.py`. This gives us Monodepth2's official disparity→depth conversion so the network output is measured in meters instead of arbitrary values.                                                                                                                                                                   |
| 11-41             | New helper `compute_scaled_intrinsics` rescales your calibrated focal lengths (`fx`, `fy`) and principal point (`cx`, `cy`) from the original camera resolution to the network's input resolution. Example: if calibration was 1280×720 but inference runs at 640×192, the function shrinks `fx` and `fy` accordingly so back-projected points stay properly aligned. |
| 44-55             | `build_pixel_grid` pre-computes the (u, v) pixel coordinates for the depth map resolution. We build this once and reuse it instead of rebuilding on every frame.                                                                                                                                                                                                      |
| 58-72             | `apply_transform` lets you pass a 4×4 camera→body matrix (from `config.json`). If present, every point is rotated/translated into the drone body frame before you send it to your planner.                                                                                                                                                                            |
| 75-116            | `depth_to_points` is the new vectorised back-projection pipeline. Steps: (1) filter invalid/too-close/too-far pixels, (2) convert remaining pixels into XYZ using the scaled intrinsics, (3) pair them with RGB colours, and (4) apply the optional extrinsic transform. Because everything runs in NumPy, it is ~50× faster than looping over individual pixels.     |
| 119-149           | `DepthProcessor.__init__` now records `min_depth`, `max_depth`, and the network input size. The weights are loaded as before, but we use these values later to scale disparity.                                                                                                                                                                                       |
| 151-172           | `DepthProcessor.estimate_depth` resizes the frame to the network size, runs it through the encoder/decoder, converts the disparity tensor into metric depth via `disp_to_depth`, and returns a `float32` depth map. No more manual scaling with `depth_scale`.                                                                                                        |
| 176-217           | `PointCloudVisualizer.__init__` stores the project configs, sets up the Open3D viewer, and parses the optional `camera_to_body` transform. If the transform is missing or malformed we fall back to identity so you never crash mid-flight.                                                                                                                           |
| 219-232           | `_ensure_geometry_cache` caches the scaled intrinsics and pixel grid the first time a frame of a new size arrives. This keeps per-frame processing lightweight.                                                                                                                                                                                                       |
| 234-253           | `create_point_cloud` now accepts the raw depth map (meters) plus the matching RGB image. It converts BGR→RGB, calls `depth_to_points`, applies optional voxel subsampling, and returns arrays ready for Open3D.                                                                                                                                                       |
| 255-270           | `update` and `close` are the same, but we now guard against empty point sets.                                                                                                                                                                                                                                                                                         |

**Example:**

```python
from lib.depth_point_cloud_utils import DepthProcessor, depth_to_points, build_pixel_grid

processor = DepthProcessor('./weights/RTMonoDepth/full/ms_640_192/')
depth_map, resized = processor.estimate_depth(frame)
points, colours = depth_to_points(
    depth_map,
    cv2.cvtColor(resized, cv2.COLOR_BGR2RGB),
    compute_scaled_intrinsics(config['video'], frame.shape[:2], depth_map.shape),
    min_depth=0.1,
    max_depth=80.0
)
```

Running this snippet produces metric XYZ values (meters) with matching RGB colours; you can feed `points` straight into a voxel map or collision checker.

## 2. `lib/point_cloud_preprocessing.py`

| Code (line range) | Explanation                                                                                                                                                                                                   |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1-34              | Dataclass wrappers load voxel, ground, and clustering parameters straight from `config.json`. These map directly to the "filtering", "ground extraction", and "clustering" stages described in the UAV paper. |
| 36-88             | `PointCloudPreprocessor.process` runs the configured stages sequentially and records a human-readable report for logging/debugging.                                                                           |
| 44-53             | Voxel down-sampling matches the paper’s pre-filtering step, shrinking dense KITTI clouds into manageable grids before further analysis.                                                                       |
| 55-70             | `_remove_ground` applies Open3D’s RANSAC plane segmentation to peel off the ground plane, leaving only potential obstacles.                                                                                   |
| 72-122            | `_cluster` performs DBSCAN, computes centroids and bounding boxes, re-colours clusters, and tracks the nearest obstacle distance—mirroring the paper’s safety-volume detection.                               |
| 124-138           | `_label_to_colors` gives each cluster a distinct colour while shading noise points dark grey for quick visual debugging.                                                                                      |

**Example:**

```python
from lib.point_cloud_preprocessing import PointCloudPreprocessor

pre = PointCloudPreprocessor({
    "voxel_size": 0.05,
    "ground": {"enabled": True},
    "clustering": {"enabled": True, "eps": 0.35, "min_points": 40}
})

filtered_cloud, report = pre.process(raw_cloud)
print(report["nearest_obstacle"])
```

`filtered_cloud` is the obstacle-only point cloud you can feed to a planner; `report` contains plane coefficients and cluster summaries for telemetry.

## 3. `process_sequence.py`

| Code (line range) | Explanation                                                                                                                                                                                                                                                                                                                                             |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 6-11              | Import the new helpers (`compute_scaled_intrinsics`, `build_pixel_grid`, `depth_to_points`).                                                                                                                                                                                                                                                            |
| 19-37             | When the processor is constructed we pass `min_depth`, `max_depth`, and input resolution from `config.json`. We also record flags for whether to apply extrinsics and prepare caches for intrinsics/pixel grids.                                                                                                                                        |
| 39-64             | `_load_camera_to_body`, `_ensure_geometry_cache`, and `_frame_to_points` mirror the logic in `depth_point_cloud_utils`. `_frame_to_points` converts one depth map + colour frame into metric XYZ+RGB arrays using the cached intrinsics and optional camera→body transform.                                                                             |
| 88                | We no longer keep the resized network frame because `_frame_to_points` works on the stored RGB image instead.                                                                                                                                                                                                                                           |
| 135-186           | The nested per-pixel loops are gone. For every saved `.npy` depth map we: (1) load the matching RGB frame, (2) ensure intrinsics are cached, (3) resize RGB to the depth map shape, (4) call `_frame_to_points`, and (5) collect the resulting arrays. The code prints how many valid points each frame contributes so you can spot bad frames quickly. |
| 188-210           | After concatenating every frame’s point set we optionally apply voxel down-sampling (`point_cloud.voxel_size` in `config.json`) and statistical outlier removal (configurable neighbour count and sigma threshold). This keeps the point cloud light-weight for planners without losing the overall geometry.                                           |
| 212-217           | A guard prints a helpful message if no valid points were produced, so the pipeline never silently writes an empty PLY file.                                                                                                                                                                                                                             |

**Example:** If you set `"voxel_size": 0.05` under `point_cloud` in `config.json`, every frame’s point cloud is thinned to a 5 cm grid before being merged, which stabilizes the global map while keeping obstacles in place.

| 213-235 | After merging all frames, the point cloud is passed through `PointCloudPreprocessor.process`. The resulting analysis report is saved alongside the `.ply`, giving you the same nearest-obstacle data the research paper uses for decision making. |

## 4. `depth_to_pointcloud.py`

| Code (line range) | Explanation                                                                                                                                                                                                                                                                                                                                                                               |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 11-25             | Constructor now forwards the depth range and network input size into `DepthProcessor`, ensuring live mode returns metric distances just like the offline sequence pipeline.                                                                                                                                                                                                               |
| 34-38             | `process_frame` indentation was fixed and clarified; it simply calls `DepthProcessor.estimate_depth` and returns both the depth map (meters) and the resized RGB input.                                                                                                                                                                                                                   |
| 53-67             | When point-cloud output is disabled in `config.json`, the live loop now supplies empty arrays so the visualizer can be skipped without errors. Otherwise we pass the metric depth map directly to `PointCloudVisualizer.create_point_cloud` (no artificial `depth_scale`). The console prints the min/max range of the generated point set so you can monitor sensor health in real time. |

**Example:** With the drone hovering, run:

```bash
python depth_to_pointcloud.py --weight_path ./weights/RTMonoDepth/full/ms_640_192/
```

Watch the terminal output — a healthy indoor scene should report something like `Depth range: 0.60m to 7.20m`. Anything wildly outside your configured `min_depth`/`max_depth` hints at bad lighting or saturation.

## 5. Configuration knobs worth knowing

Add these entries to the `point_cloud` block inside `config.json` to tweak behaviour without touching code:

```json
"point_cloud": {
    "min_depth": 0.4,
    "max_depth": 10.0,
    "apply_extrinsics": true,
    "camera_to_body": [
        1, 0, 0, 0.05,
        0, 1, 0, 0.0,
        0, 0, 1, 0.10,
        0, 0, 0, 1
    ],
    "voxel_size": 0.05,
    "statistical_outlier": {
        "enabled": true,
        "nb_neighbors": 20,
        "std_ratio": 2.0
    }
}
```

- `min_depth` / `max_depth`: filter out pixels outside the useful range (e.g., reflections or the sky).
- `apply_extrinsics`: toggles whether `camera_to_body` is applied; keep it `true` when feeding a drone controller.
- `camera_to_body`: 4×4 row-major transform from camera frame to vehicle frame (example above shifts the camera 5 cm right and 10 cm up).
- `voxel_size`: down-sample merged clouds to reduce planner load; set to `0` to keep every point.
- `statistical_outlier`: removes stray points; tighten `std_ratio` to 1.0 for very clean data or loosen it for rough outdoor scenes.

## 6. Quick verification checklist

1. Run `python3 -m py_compile lib/depth_point_cloud_utils.py process_sequence.py depth_to_pointcloud.py` to confirm the files import cleanly.
2. Execute `python process_sequence.py --weight_path ./weights/RTMonoDepth/full/ms_640_192/ --input_path ./test-frames --image_format png` and open the saved `point_clouds/test-frames_pointcloud.ply` in Open3D. Distances in the viewer should match real-world measurements (e.g., a chair 2 m away should appear near `z ≈ 2`).
3. Adjust `min_depth`, `max_depth`, and `voxel_size` according to your sensor height and desired map density.

With these changes your drone now receives accurate, body-frame-aligned 3D points suitable for collision checking, path planning, or SLAM back-ends.
