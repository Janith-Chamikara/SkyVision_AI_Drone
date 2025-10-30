# Beginner-Friendly Guide: From Single Camera Frames to Obstacle-Ready Point Clouds

This document explains every stage of the updated RT-MonoDepth pipeline in plain language. Even if you are new to computer vision or robotics, you will learn what happens to each image, why those steps matter, and the key assumptions behind them.

---

## 1. Starting Point: What Data Do We Have?

- **Input**: A folder of RGB images from a forward-facing monocular camera mounted on the UAV/drone.
- **Goal**: Produce a 3D point cloud where each point has a position (X, Y, Z in meters) and colour. The cloud should highlight obstacles and ignore the ground so a planner can decide where it is safe to fly.

### Core Assumptions

1. **Camera Intrinsics are known** – We know the focal lengths (`fx`, `fy`) and the principal point (`cx`, `cy`) thanks to camera calibration (e.g., OpenCV chessboard calibration). These values live in `config.json` under `video`.
2. **Camera Extrinsics relative to the drone body are known or approximated** – If you want points in the drone frame, you provide a 4×4 `camera_to_body` transform in `config.json`. If you skip it, we assume the camera frame is the same as the body frame.
3. **Ground is approximately planar** – Our ground-removal step uses a single plane model. This works indoors and on relatively flat outdoor areas. If the surface is highly uneven, you may need a more advanced filter.
4. **Monocular depth network is trained for the environment** – RT-MonoDepth is pre-trained on outdoor driving scenes. It generalises surprisingly well indoors, but the metric scale is best when the environment matches the training data.

---

## 2. Turning RGB Pixels into Depth (Meters)

**Relevant code**: `lib/depth_point_cloud_utils.py` → `DepthProcessor`

1. **Resizing and Normalisation**: Each image is resized to the network’s expected input size (default 640×192) and scaled to `[0, 1]`.
2. **Depth Prediction**: RT-MonoDepth outputs a _disparity_ map (values between 0 and 1). Disparity is inversely proportional to depth: large disparity = close object.
3. **Disparity to Depth Conversion**: We apply Monodepth2’s `disp_to_depth` formula:

   \[
   d = \frac{1}{ \frac{1}{d*{\text{max}}} + \left( \frac{1}{d*{\text{min}}} - \frac{1}{d\_{\text{max}}} \right) \cdot \text{disp} }
   \]

   where `d_min` and `d_max` come from `config.json` (`point_cloud.min_depth/max_depth`). The result is a metric depth map in meters.

**Beginner takeaway**: The neural network estimates how far each pixel is from the camera. We convert those estimates into real-world distances using an equation from the original research paper.

---

## 3. Back-Projecting Depth into 3D Points

**Relevant code**: `depth_to_points` in `lib/depth_point_cloud_utils.py`

1. **Scale Intrinsics**: If the calibration was done at a different resolution than the network input, we scale (`fx`, `fy`, `cx`, `cy`) to the depth map size. This keeps math consistent.
2. **Meshgrid of Pixels**: We create arrays of pixel coordinates `(u, v)` for every pixel in the depth map.
3. **3D Conversion**: Using the pinhole camera model:

   \[
   Z = \text{depth}(u, v) \\
   X = (u - c_x) \cdot Z / f_x \\
   Y = (v - c_y) \cdot Z / f_y
   \]

   This converts each depth pixel into a 3D point `(X, Y, Z)`.

4. **Optional Body Transform**: If `apply_extrinsics` is true, we multiply the points by the provided 4×4 matrix to express them in the drone’s frame.
5. **Attach Colours**: We take the RGB value from the same pixel and attach it to the 3D point.

**Assumptions**: Lens distortion is negligible or already rectified; the camera obeys the pinhole model.

---

## 4. Combining Frames into a Single Point Cloud

**Relevant code**: `process_sequence.py`

1. For each depth map, we convert it to a set of coloured 3D points.
2. We append them to a master list to merge all frames into a single dense point cloud.
3. Optional initial clean-up:
   - **Voxel Down-sample** (optional) – Shrinks the point cloud to a regular 3D grid to reduce density.
   - **Statistical Outlier Removal** (optional) – Removes isolated points likely caused by noise.

**Why merge frames?** A single monocular depth map is sparse along the viewing direction. Merging many frames increases coverage and fills in occluded areas.

---

## 5. Preprocessing for Obstacle Avoidance

**Relevant code**: `lib/point_cloud_preprocessing.py`

This mirrors the workflow diagram in the UAV research paper.

### 5.1 Filtering / Down-sampling (optional but recommended)

- `voxel_size` in `config.json` controls the 3D grid size (meters).
- Example: `0.05` means we keep at most one point per 5 cm cube.
- Benefits: speeds up later processing and smooths noisy surfaces.

### 5.2 Ground Removal (RANSAC plane fitting)

- **Goal**: Separate ground points from obstacles.
- We run `segment_plane` on the full point cloud. RANSAC repeatedly samples three points, fits a plane, and counts how many nearby points (within `distance_threshold`) agree.
- The plane with most inliers is assumed to be the ground.
- Points belonging to this plane are removed, leaving only “non-ground” points.
- **Assumption**: The ground is roughly flat and represented by a single dominant plane.

### 5.3 Segmentation & Clustering (DBSCAN)

- DBSCAN groups neighbouring points into clusters without predefining the number of clusters.
- `eps` controls how close points must be to be considered part of the same cluster.
- `min_points` sets the minimum cluster size.
- Each cluster becomes a candidate obstacle; noise points (label `-1`) are ignored.

### 5.4 Bounding Boxes and Nearest Obstacle

- For each cluster we compute:
  - centroid (average XYZ)
  - axis-aligned bounding box (min/max corners & size)
  - distance from the origin (drone) to the centroid
- We track the nearest cluster. When integrating with a planner, this acts like the “safety volume” detector from the paper.

### 5.5 Report Generation (JSON)

- After processing, we write a report to `point_clouds/<sequence>_analysis.json` with:
  - Each preprocessing step and point counts
  - Ground plane coefficients `[a, b, c, d]` for the plane equation `ax + by + cz + d = 0`
  - Clusters, bounding boxes, centroids, nearest obstacle distance

**Example report snippet**:

```json
{
  "steps": [
    {
      "name": "voxel_downsample",
      "voxel_size": 0.05,
      "point_count": 8975
    },
    {
      "name": "ground_removal",
      "point_count": 6420,
      "plane_model": [0.01, -0.99, 0.05, -1.2],
      "ground_points": 2555
    },
    {
      "name": "clustering",
      "clusters": [
        {
          "label": 0,
          "points": 1500,
          "centroid": [1.2, 0.3, 4.1],
          "extent": [0.9, 0.5, 1.2]
        }
      ],
      "max_label": 2,
      "noise_points": 120,
      "nearest_obstacle": {
        "label": 1,
        "distance": 2.35,
        "centroid": [0.8, -0.2, 2.2]
      }
    }
  ],
  "ground_plane": [0.01, -0.99, 0.05, -1.2],
  "nearest_obstacle": {
    "label": 1,
    "distance": 2.35,
    "centroid": [0.8, -0.2, 2.2]
  }
}
```

---

## 6. Live Viewer (`depth_to_pointcloud.py`)

When you run:

```bash
python depth_to_pointcloud.py --weight_path ./weights/RTMonoDepth/full/ms_640_192/
```

- We grab frames from the webcam, estimate depth, build a point cloud, run the same preprocessing steps, and show the filtered result in Open3D.
- The console prints the nearest cluster distance so you immediately know if something is too close to the drone.

---

## 7. Practical Tips & Tunable Parameters

| Parameter                   | Where                                        | Meaning                                              | Typical Range                            |
| --------------------------- | -------------------------------------------- | ---------------------------------------------------- | ---------------------------------------- |
| `min_depth`, `max_depth`    | `config.json` (`point_cloud`)                | Depth bounds fed into disparity-to-depth conversion. | Indoors: 0.3 – 8.0; Outdoors: 0.3 – 80.0 |
| `voxel_size`                | `point_cloud` or `point_cloud_preprocessing` | Grid cell size for down-sampling.                    | 0.02 – 0.1 (meters)                      |
| `ground.distance_threshold` | `point_cloud_preprocessing.ground`           | Allowed distance to the ground plane.                | 0.02 – 0.1                               |
| `clustering.eps`            | `point_cloud_preprocessing.clustering`       | Radius for DBSCAN neighbours.                        | 0.2 – 0.6                                |
| `clustering.min_points`     | same as above                                | Minimum points per cluster.                          | 20 – 200                                 |

### Choosing Values

- **Smaller voxel sizes** retain detail but increase computation.
- **Smaller ground thresholds** make ground removal strict (good for flat floors), but may misclassify gentle slopes.
- **Larger `eps`** merges nearby clusters—useful outdoors where obstacles are big.

---

## 8. End-to-End Workflow Summary

1. **Run depth estimation**: `process_sequence.py` reads RGB frames, writes per-frame depth maps, and generates a depth video for visual inspection.
2. **Generate obstacle cloud**: The same script merges depth maps, filters the point cloud, removes ground, clusters obstacles, and saves both `.ply` and `.json` analysis files.
3. **Inspect results**:
   - Open the `.ply` file in Open3D or MeshLab to view the obstacle-only cloud.
   - Open the `.json` report to see nearest obstacle distance and cluster statistics.
4. **Online mode**: `depth_to_pointcloud.py` runs the full pipeline live from a webcam and prints obstacle distances in real time.

---

## 9. Theories in Play

- **Pinhole Camera Model**: Relates pixel coordinates to 3D rays using intrinsic parameters.
- **Monocular Depth Estimation**: Neural network predicts depth from a single RGB image using learned priors.
- **Plane Fitting with RANSAC**: Robust method to find a dominant plane even when many points are outliers.
- **Density-Based Clustering (DBSCAN)**: Groups nearby points without needing to specify the number of clusters in advance.

Understanding these theories helps you tweak parameters confidently and extend the pipeline (e.g., replacing DBSCAN with a different segmentation method).

---

## 10. What to Do Next

- **Test different environments**: tweak `min_depth`, `max_depth`, `voxel_size`, `eps`, and `distance_threshold` until the clusters match real obstacles.
- **Integrate with a planner**: feed the nearest obstacle distance or the full clustered cloud into your avoidance logic.
- **Expand the analysis**: extend the JSON report with additional metrics like obstacle velocity if you add temporal tracking.

With this guide, you should now understand—not just run—the upgraded pipeline. You have metric point clouds, ground-free obstacle clusters, and a live-ready preprocessing stage that mirrors the research literature.
