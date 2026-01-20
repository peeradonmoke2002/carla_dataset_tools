# Range-Based Labeling for Stable Auto-Labeling

## Overview

This repository now supports two labeling modes:

1. **Range-Based Filtering** (Recommended - Stable)
2. **Point Cloud-Based Filtering** (Traditional - Less Stable)

## Why Range-Based is More Stable

### Traditional Point Cloud-Based Approach (Unstable)
- Requires minimum LiDAR points in bounding box (5 for pedestrians, 20 for cars)
- Point density varies with:
  - Distance from sensor
  - Occlusion from other objects
  - Vehicle/pedestrian orientation
  - LiDAR sensor noise and dropouts
- Small pedestrians may have <5 points even when clearly visible
- Results in inconsistent labeling

### Range-Based Approach (Stable)
- Uses rectangular spatial boundary check (±51.2m in X and Y by default)
- Based on CARTI_Dataset methodology
- Labels objects using ground truth transforms from CARLA
- No dependency on LiDAR point cloud quality
- Consistent results regardless of:
  - Sensor noise
  - Object orientation
  - Occlusion levels

## Configuration

Edit the parameters in `carla_dataset_tools/label_tools/kitti_object/kitti_object_helper.py`:

```python
class Param:
    # ... other parameters ...

    # Range-based filtering configuration
    USE_RANGE_BASED_FILTER = True  # Set to False for point cloud-based mode
    RANGE_BOX_SIZE = 51.2  # Detection range in meters (±51.2m in X and Y)
```

## How It Works

### Range-Based Mode (USE_RANGE_BASED_FILTER = True)

1. Checks if object center is within rectangular box:
   - X: [sensor.x - 51.2, sensor.x + 51.2]
   - Y: [sensor.y - 51.2, sensor.y + 51.2]
   - Z: No restriction (all heights)

2. If within range:
   - Label is generated from CARLA ground truth
   - Occlusion set to 0 (fully visible)
   - No point cloud validation required

3. Then applies standard 2D projection filters:
   - Vertex visibility in image
   - Minimum bbox dimensions
   - Truncation calculation

### Point Cloud-Based Mode (USE_RANGE_BASED_FILTER = False)

1. Checks Euclidean distance (1.0m - 150.0m)
2. Validates minimum LiDAR points in bounding box:
   - Cars: ≥20 points
   - Pedestrians: ≥5 points (with 2x scaled bbox)
3. Calculates occlusion level from point count
4. Then applies 2D projection filters

## Recommended Settings

### For Stable Dataset Generation (Recommended)
```python
USE_RANGE_BASED_FILTER = True
RANGE_BOX_SIZE = 51.2  # Cooperative perception range (CARTI standard)
```

### For Realistic LiDAR Simulation
```python
USE_RANGE_BASED_FILTER = False
POINTS_MIN_CAR = 20
POINTS_MIN_PEDESTRIAN = 5
```

### For Extended Range Detection
```python
USE_RANGE_BASED_FILTER = True
RANGE_BOX_SIZE = 75.0  # Increased to 75 meters
```

## Usage

Simply run the labeling tool as usual. The mode is automatically selected based on the `USE_RANGE_BASED_FILTER` parameter:

```bash
python carla_dataset_tools/label_tools/kitti_objects_label.py \
    --record record_2024_0120_1234 \
    --vehicle vehicle.tesla.model3_1 \
    --lidar velodyne \
    --camera image_2
```

## Reference

This implementation is based on the CARTI_Dataset approach:
- Paper: "CARTI: Cooperative Autonomous Road Transport Infrastructure"
- Reference: `/home/peeradon/CARTI_Dataset/CARTI_Dataset_V1.0.py`
- Key functions: `isInRange()` (lines 92-106)

## Comparison

| Aspect | Range-Based | Point Cloud-Based |
|--------|-------------|-------------------|
| Stability | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Realism | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Speed | ⭐⭐⭐⭐⭐ (faster) | ⭐⭐⭐ |
| Completeness | ⭐⭐⭐⭐⭐ (more labels) | ⭐⭐⭐ |
| Use Case | Training datasets | LiDAR simulation testing |

## Files Modified

1. `carla_dataset_tools/label_tools/kitti_object/kitti_object_helper.py`
   - Added `Param.USE_RANGE_BASED_FILTER`
   - Added `Param.RANGE_BOX_SIZE`
   - Added `is_in_range_box()` function

2. `carla_dataset_tools/label_tools/kitti_objects_label.py`
   - Updated `process_frame()` to support both modes
   - Conditional logic for range-based vs point cloud-based filtering
