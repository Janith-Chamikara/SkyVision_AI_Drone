# Autonomous Navigation System - Setup Guide

This guide walks you through setting up and running the complete autonomous drone navigation system.

## 📋 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  THREAD 1: Camera (30-60 Hz)                 │
│  Video Capture → Depth Estimation → Point Cloud Generation  │
└────────────────────────┬────────────────────────────────────┘
                         │ Point Cloud Buffer
┌────────────────────────▼────────────────────────────────────┐
│              THREAD 2: Navigation (30 Hz)                    │
│  Point Cloud Filtering → Occupancy Grid → VFH+ → Commands   │
└────────────────────────┬────────────────────────────────────┘
                         │ Velocity Commands
┌────────────────────────▼────────────────────────────────────┐
│                THREAD 3: Motors (50 Hz)                      │
│        Velocity → Motor Mixing → UART Transmission          │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade

# Install Python dependencies
pip install -r requirements.txt

# For Jetson: Install CUDA-accelerated libraries
sudo apt-get install python3-opencv
```

### 2. Download Model Weights

Download RTMonoDepth weights from [Google Drive](https://drive.google.com/file/d/1Jf5K3m0DfAqVcVCE6y0cKufEKIHu86sz/view?usp=drive_link) and place in `weights/RTMonoDepth/full/ms_640_192/`.

### 3. Run Simulation Mode (Safe Testing)

```bash
# Test with webcam
python autonomous_navigation_system.py \
  --weight_path ./weights/RTMonoDepth/full/ms_640_192/ \
  --config config_navigation.json \
  --source 0

# Test with video file
python autonomous_navigation_system.py \
  --weight_path ./weights/RTMonoDepth/full/ms_640_192/ \
  --config config_navigation.json \
  --source path/to/video.mp4

# Test with RTSP stream
python autonomous_navigation_system.py \
  --weight_path ./weights/RTMonoDepth/full/ms_640_192/ \
  --config config_navigation.json \
  --source rtsp://192.168.1.100:8554/stream
```

### 4. Enable Hardware Control (⚠️ ONLY when drone is secured!)

```bash
python autonomous_navigation_system.py \
  --weight_path ./weights/RTMonoDepth/full/ms_640_192/ \
  --config config_navigation.json \
  --source rtsp://DRONE_IP:8554/stream \
  --enable_motors
```

## ⚙️ Configuration

Edit `config_navigation.json` to tune parameters:

### Key Parameters

**Point Cloud Processing:**
- `voxel_size`: 0.05m (5cm voxels) - reduces 500k → 50-100k points
- `nb_neighbors`: 20 neighbors for outlier detection
- `std_ratio`: 2.0 standard deviations threshold
- `temporal_buffer_size`: 3 frames for smoothing

**Occupancy Grid:**
- `grid_resolution`: 0.1m cells (10cm resolution)
- `grid_size_x/y/z`: 10m x 10m x 3m grid
- `occupancy_threshold`: 0.5 (50% confidence)
- `decay_rate`: 0.9 per frame

**Navigation (VFH+):**
- `histogram_resolution`: 5° angular sectors (72 sectors total)
- `obstacle_threshold`: 0.3 density threshold
- `max_velocity`: 2.0 m/s
- `max_angular_velocity`: 1.57 rad/s (90°/s)
- `safety_distance`: 1.0m minimum clearance
- `control_frequency`: 30 Hz

**Motor Control:**
- `port`: /dev/ttyTHS1 (Jetson UART)
- `control_rate`: 50 Hz
- `max_tilt_angle`: 30° pitch/roll limit
- `hover_throttle`: 128 (mid-range PWM)

## 🎮 Controls

While the system is running:

- **Q** or **ESC** - Emergency stop and quit
- **P** - Pause/Resume
- **R** - Reset navigation
- **Arrow Keys** - Set goal direction
  - Up: Forward
  - Left: 45° left
  - Right: 45° right
- **S** - Show/hide detailed statistics

## 📊 Performance Monitoring

### Target Performance (Jetson Xavier NX)

| Component | Target | Acceptable |
|-----------|--------|------------|
| Camera FPS | 30-60 Hz | > 20 Hz |
| Navigation FPS | 30 Hz | > 20 Hz |
| Motor Control | 50 Hz | > 30 Hz |
| GPU Utilization | 70-85% | < 90% |
| CPU Load | < 60% | < 80% |
| Temperature | < 70°C | < 80°C |

### Monitor with tegrastats

```bash
# Terminal 1: Run tegrastats
tegrastats

# Terminal 2: Run navigation
python autonomous_navigation_system.py ...
```

Look for:
- **GR3D**: GPU utilization (aim for 70-85%)
- **CPU**: CPU load per core (aim for < 60% average)
- **temp**: Board temperature (< 70°C)

## 🔧 Calibration

### Camera Intrinsics

Update `config_navigation.json` with your camera parameters:

```json
"video": {
  "fx": YOUR_FOCAL_X,
  "fy": YOUR_FOCAL_Y,
  "cx": YOUR_PRINCIPAL_X,
  "cy": YOUR_PRINCIPAL_Y
}
```

Get calibration using:
```bash
python cam_calibration/calibrate.py --images calibration_images/
```

### Motor Gains

Tune motor mixing gains in `config_navigation.json`:

```json
"motor_control": {
  "velocity_to_pitch_gain": 0.1,   # Adjust for desired responsiveness
  "angular_velocity_to_yaw_gain": 30,
  "hover_throttle": 128             # Adjust for your drone's weight
}
```

## 🧪 Testing Procedure

### Phase 1: Simulation Testing
1. ✅ Test with webcam - walk in front of camera
2. ✅ Test with video file - verify obstacle avoidance
3. ✅ Test with Gazebo simulation
4. ✅ Verify FPS is > 20 Hz for all components

### Phase 2: Hardware-in-Loop (No Motors)
1. ✅ Connect to drone's camera via RTSP
2. ✅ Verify point cloud quality
3. ✅ Test obstacle detection range
4. ✅ Validate navigation commands (logged only)

### Phase 3: Motor Control (Secured Drone)
1. ⚠️ Secure drone on test stand
2. ⚠️ Enable motors with `--enable_motors`
3. ⚠️ Verify motor responses are correct
4. ⚠️ Test emergency stop (Q key)

### Phase 4: Flight Testing
1. ⚠️ Start with low `max_velocity` (0.5 m/s)
2. ⚠️ Test in open area with no obstacles first
3. ⚠️ Gradually increase velocity and add obstacles
4. ⚠️ Always have manual override ready

## 🐛 Troubleshooting

### Low FPS
- Reduce `voxel_size` (e.g., 0.1m instead of 0.05m)
- Increase `subsample_factor` (e.g., 8 instead of 4)
- Disable `compute_distance_transform` in occupancy grid
- Use `--no_cuda` if GPU is overloaded

### Poor Obstacle Detection
- Increase `max_detection_range`
- Decrease `obstacle_threshold` (more sensitive)
- Increase `temporal_buffer_size` (smoother but laggier)
- Calibrate camera intrinsics

### Erratic Navigation
- Increase `valley_min_width` (require wider safe passages)
- Decrease `max_velocity` and `max_angular_velocity`
- Increase `safety_distance`
- Tune `histogram_resolution` (try 10° for fewer sectors)

### Serial Communication Errors
- Check port: `ls /dev/ttyTHS*` or `ls /dev/ttyUSB*`
- Verify baudrate matches flight controller
- Add user to dialout group: `sudo usermod -a -G dialout $USER`
- Test with: `python -m serial.tools.miniterm /dev/ttyTHS1 115200`

## 📁 Project Structure

```
SkyVision_AI_Drone/
├── autonomous_navigation_system.py    # Main integrated system
├── config_navigation.json             # Navigation configuration
├── requirements.txt                   # Python dependencies
├── lib/
│   ├── depth_point_cloud_utils.py    # Depth → point cloud conversion
│   ├── video_utils.py                # Video capture utilities
│   ├── processing/
│   │   ├── __init__.py
│   │   └── point_cloud_processor.py  # Step 1: Filtering & downsampling
│   ├── navigation/
│   │   ├── __init__.py
│   │   ├── occupancy_grid.py         # Step 2: Spatial representation
│   │   └── vfh_plus.py               # Step 3: Obstacle avoidance
│   └── control/
│       ├── __init__.py
│       └── motor_controller.py       # Step 4: Motor commands
├── networks/
│   └── RTMonoDepth/                  # Depth estimation models
└── weights/                           # Model weights
```

## 🔐 Safety Notes

1. **Always test in simulation first**
2. **Use emergency stop (Q key) liberally**
3. **Start with low velocities**
4. **Keep manual override ready**
5. **Test in open areas first**
6. **Monitor system temperature**
7. **Have multiple observers during flight tests**

## 📈 Performance Timing Breakdown

Typical per-cycle timing on Jetson Xavier NX:

| Component | Time | Notes |
|-----------|------|-------|
| Depth estimation | 15-20ms | RTMonoDepth inference |
| Point cloud generation | 2-3ms | Pinhole projection |
| Voxel downsampling | 2-3ms | 500k → 50k points |
| Outlier removal | 3-5ms | Statistical filtering |
| Occupancy grid update | 1-2ms | 3D discretization |
| VFH+ computation | 3-5ms | Polar histogram |
| **Total (Navigation Thread)** | **~30ms** | **33 Hz achievable** |

## 🆘 Support

For issues:
1. Check `tegrastats` output for resource bottlenecks
2. Review logs for error messages
3. Test each component individually
4. Verify camera calibration
5. Check serial port permissions

## 📚 References

- RTMonoDepth: Real-time monocular depth estimation
- VFH+: Vector Field Histogram for obstacle avoidance
- Open3D: Point cloud processing library
- Jetson: NVIDIA embedded AI platform
