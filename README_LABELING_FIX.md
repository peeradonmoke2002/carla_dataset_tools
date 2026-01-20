# Auto-Labeling Stability Fix

## Summary

This repository has been updated with **stable range-based labeling** inspired by the CARTI_Dataset approach. This fixes the instability issues with point cloud-based labeling.

## What Changed?

### Before (Unstable)
- Labels were generated only if objects had enough LiDAR points in their bounding box
- Pedestrians needed ≥5 points, cars needed ≥20 points
- Many valid objects were missed due to:
  - Sensor noise and dropouts
  - Occlusion by other objects
  - Distance from sensor
  - Object orientation

### After (Stable)
- Labels are generated for all objects within a rectangular detection range
- Uses ±51.2m box in X and Y directions (configurable)
- Based on ground truth transforms from CARLA, not point cloud
- No dependency on LiDAR point density
- Consistent and complete labeling

## Quick Start

### Check Current Configuration
```bash
python3 configure_labeling_mode.py --status
```

### Switch to Range-Based Mode (Recommended)
```bash
python3 configure_labeling_mode.py --mode range
```

### Switch to Point Cloud Mode (Traditional)
```bash
python3 configure_labeling_mode.py --mode pointcloud
```

### Adjust Detection Range
```bash
# Set to 75 meters
python3 configure_labeling_mode.py --range-size 75.0

# Set to 100 meters
python3 configure_labeling_mode.py --range-size 100.0
```

## Usage

After configuring the mode, simply run your labeling as usual:

```bash
python carla_dataset_tools/label_tools/kitti_objects_label.py \
    --record record_2024_0120_1234 \
    --vehicle vehicle.tesla.model3_1 \
    --lidar velodyne \
    --camera image_2
```

The system will automatically use the configured mode.

## Files Modified

1. **[kitti_object_helper.py](carla_dataset_tools/label_tools/kitti_object/kitti_object_helper.py)**
   - Added `Param.USE_RANGE_BASED_FILTER` (default: `True`)
   - Added `Param.RANGE_BOX_SIZE` (default: `51.2` meters)
   - Added `is_in_range_box()` function for stable range checking

2. **[kitti_objects_label.py](carla_dataset_tools/label_tools/kitti_object/kitti_objects_label.py)**
   - Updated `process_frame()` method
   - Conditional logic for range-based vs point cloud-based filtering
   - Lines 134-175: New dual-mode implementation

3. **[configure_labeling_mode.py](configure_labeling_mode.py)** (NEW)
   - Configuration helper script
   - Easy mode switching
   - Status display

4. **[RANGE_BASED_LABELING.md](RANGE_BASED_LABELING.md)** (NEW)
   - Detailed documentation
   - Comparison of approaches
   - Technical details

## Comparison

| Aspect | Range-Based (NEW) | Point Cloud-Based (OLD) |
|--------|------------------|------------------------|
| **Stability** | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Completeness** | ⭐⭐⭐⭐⭐ More labels | ⭐⭐⭐ Misses objects |
| **Speed** | ⭐⭐⭐⭐⭐ Faster | ⭐⭐⭐ Slower |
| **Realism** | ⭐⭐⭐ Uses ground truth | ⭐⭐⭐⭐⭐ Uses LiDAR |
| **Use Case** | Training datasets | LiDAR testing |

## Reference

Based on CARTI_Dataset methodology:
- Repository: `/home/peeradon/CARTI_Dataset`
- Paper: "CARTI: Cooperative Autonomous Road Transport Infrastructure"
- Key function: `isInRange()` in `CARTI_Dataset_V1.0.py` (lines 92-106)

## Manual Configuration

If you prefer to configure manually, edit [kitti_object_helper.py](carla_dataset_tools/label_tools/kitti_object/kitti_object_helper.py):

```python
class Param:
    # ... other parameters ...

    # For stable labeling (recommended)
    USE_RANGE_BASED_FILTER = True
    RANGE_BOX_SIZE = 51.2

    # For traditional point cloud validation
    # USE_RANGE_BASED_FILTER = False
    # POINTS_MIN_CAR = 20
    # POINTS_MIN_PEDESTRIAN = 5
```

## Troubleshooting

### "Too many labels generated"
- The range-based mode labels ALL objects in range
- If you want fewer labels, reduce `RANGE_BOX_SIZE`:
  ```bash
  python3 configure_labeling_mode.py --range-size 30.0
  ```

### "Want more realistic LiDAR simulation"
- Switch to point cloud-based mode:
  ```bash
  python3 configure_labeling_mode.py --mode pointcloud
  ```

### "Missing pedestrians in labels"
- If using point cloud mode, pedestrians may be filtered out
- Switch to range-based mode for complete pedestrian coverage:
  ```bash
  python3 configure_labeling_mode.py --mode range
  ```

## Support

For detailed technical information, see [RANGE_BASED_LABELING.md](RANGE_BASED_LABELING.md).
