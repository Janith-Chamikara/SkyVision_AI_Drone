# RT-MonoDepth Usage Guide

This guide explains how to use the RT-MonoDepth project for depth estimation, video visualization, and point cloud generation.

## Project Structure

```
RT-MonoDepth/
├── depth_videos/      # Output directory for depth visualization videos
├── point_clouds/      # Output directory for generated point clouds
├── kitti-frames/      # Example input frames
├── weights/           # Model weights
└── test-images/       # Test images for single image processing
```

## Main Scripts

### 1. Process Image Sequence (Video + Point Cloud)
Use `process_sequence.py` to process a sequence of images, generate depth maps, create a visualization video, and build a point cloud.

```bash
python process_sequence.py --weight_path ./weights/RTMonoDepth/full/ms_640_192/ --input_path ./kitti-frames --image_format png
```

This will:
- Show real-time visualization of RGB and depth maps
- Save depth visualization video in `depth_videos/` folder
- Generate and save point cloud in `point_clouds/` folder
- Display the final point cloud in an interactive 3D viewer

**Options:**
- `--weight_path`: Path to model weights directory (required)
- `--input_path`: Path to input image sequence directory (required)
- `--image_format`: Input image format (default: 'jpg')
- `--config`: Path to configuration file (default: 'config.json')
- `--no_cuda`: Disable CUDA if available

### 2. Single Image Depth Estimation
Use `image_to_depth_full.py` for processing individual images:

```bash
python image_to_depth_full.py --image_path ./test-images/1.jpg --weight_path ./weights/RTMonoDepth/full/ms_640_192/
```

This will:
- Generate depth map for the input image
- Display RGB and depth visualization
- Save depth map outputs in the same directory as input image

### 3. Generate Point Cloud from Image
Use `depth_to_pointcloud.py` to create a point cloud from a single image:

```bash
python depth_to_pointcloud.py --image_path ./test-images/1.jpg --weight_path ./weights/RTMonoDepth/full/ms_640_192/
```

This will:
- Generate depth map
- Create and display 3D point cloud
- Save point cloud file in PLY format

## Visualization Controls

### Point Cloud Viewer
- Left mouse button: Rotate view
- Middle mouse button: Pan
- Right mouse button: Zoom
- '[' and ']': Change point size
- 'r': Reset view
- 'q': Close viewer

### Video Visualization
- RGB image shown on the left
- Depth map visualization on the right (MAGMA colormap)
- Frame counter displayed in top-left corner
- Press 'Q' to quit visualization

## Output Files

### Video Output
- Location: `depth_videos/` directory
- Format: MP4
- Content: Side-by-side RGB and depth visualization
- Filename pattern: `{input_folder_name}_depth.mp4`

### Point Cloud Output
- Location: `point_clouds/` directory
- Format: PLY
- Content: 3D point cloud with RGB colors
- Filename pattern: `{input_folder_name}_pointcloud.ply`

## Examples

1. Process KITTI sequence:
```bash
python process_sequence.py --weight_path ./weights/RTMonoDepth/full/ms_640_192/ --input_path ./kitti-frames --image_format png
```

2. Process single image:
```bash
python image_to_depth_full.py --image_path ./test-images/1.jpg --weight_path ./weights/RTMonoDepth/full/ms_640_192/
```

3. Generate point cloud from image:
```bash
python depth_to_pointcloud.py --image_path ./test-images/1.jpg --weight_path ./weights/RTMonoDepth/full/ms_640_192/
```

## Notes
- Ensure all required directories (`depth_videos/`, `point_clouds/`) exist before running scripts
- Point clouds are automatically cleaned using statistical outlier removal
- Videos are saved at 30 FPS with side-by-side visualization
- For best performance, use CUDA-enabled GPU if available