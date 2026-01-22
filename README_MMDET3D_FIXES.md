# MMDetection3D Fixes for KITTI/CARLA Dataset Training

This document describes the fixes applied to mmdetection3d (Tier4 fork) to enable CenterPoint training on KITTI-format datasets generated from CARLA simulator.

## Summary of Issues and Fixes

### 1. NumPy Compatibility Fix (`np.long` deprecated)

**File:** `mmdet3d/datasets/transforms/dbsampler.py` (Line 283)

**Error:**
```
AttributeError: module 'numpy' has no attribute 'long'
```

**Cause:** NumPy deprecated `np.long` in newer versions.

**Fix:** Changed `np.long` to `np.int64`

```python
# Before:
dtype=np.long

# After:
dtype=np.int64
```

---

### 2. Missing Velocity Handling for KITTI Dataset

**File:** `mmdet3d/models/dense_heads/centerpoint_head.py` (Lines 565-571)

**Error:**
```
ValueError: not enough values to unpack (expected 2, got 0)
```

**Cause:** CenterPoint was designed for NuScenes which includes velocity (vx, vy) in bounding boxes (9 parameters: x, y, z, w, l, h, yaw, vx, vy). KITTI format only has 7 parameters (no velocity).

**Fix:** Added conditional check to handle boxes without velocity:

```python
# Before:
vx, vy = task_boxes[idx][k][7:]

# After:
if task_boxes[idx][k].shape[0] >= 9:
    vx, vy = task_boxes[idx][k][7:]
else:
    # KITTI format doesn't have velocity, use zeros
    vx = torch.tensor(0.0, device=device)
    vy = torch.tensor(0.0, device=device)
```

---

### 3. DontCare Label Handling in KITTI Metric

**File:** `mmdet3d/evaluation/metrics/kitti_metric.py` (Lines 127-155)

**Error:**
```
KeyError: -1
```

**Cause:** KITTI dataset uses label `-1` for "DontCare" objects which should be ignored during evaluation. The metric code tried to look up this label in the class mapping dictionary.

**Fix:** Added skip condition for invalid labels and proper empty array handling:

```python
# In the annotation conversion loop:
for instance in annos['instances']:
    label = instance['bbox_label']
    # Skip DontCare or invalid labels (label=-1)
    if label < 0 or label not in label2cat:
        continue
    kitti_annos['name'].append(label2cat[label])
    # ... rest of fields

# Handle case where all instances were skipped (empty annotations):
if len(kitti_annos['name']) == 0:
    kitti_annos = {
        'name': np.array([]),
        'truncated': np.array([]),
        'occluded': np.array([]),
        'alpha': np.array([]),
        'bbox': np.zeros([0, 4]),
        'dimensions': np.zeros([0, 3]),
        'location': np.zeros([0, 3]),
        'rotation_y': np.array([]),
        'score': np.array([]),
    }
```

---

## Configuration Adjustments

### Grid Size Must Be Divisible by 8

**Error:**
```
RuntimeError: The size of tensor a (55) must match the size of tensor b (50) at non-singleton dimension 3
```

**Cause:** SECONDFPN backbone requires grid dimensions to be divisible by 8.

**Solution:** Adjusted voxel size and point cloud range:
- Voxel size: `[0.2, 0.2, 4]`
- Point cloud range: `[0, -40, -3, 72, 40, 1]`
- Grid size: `[360, 400, 1]` (360/8=45, 400/8=50 ✓)

### Validation Segmentation Fault Workaround

**Issue:** KITTI evaluation causes segmentation fault (likely numba JIT compilation issue).

**Workaround:** Set `val_interval=100` in training config to skip validation during training. Run evaluation manually after training:

```bash
python tools/test.py <config> <checkpoint>
```

---

## Custom Config File

Created: `configs/centerpoint/centerpoint_pillar02_second_secfpn_8xb4-cyclic-20e_kitti-3d.py`

Key settings:
- **Classes:** Car, Pedestrian (2 classes for CARLA dataset)
- **Voxel size:** [0.2, 0.2, 4]
- **Point cloud range:** [0, -40, -3, 72, 40, 1]
- **Grid size:** [360, 400, 1]
- **Epochs:** 80
- **Batch size:** 4
- **Checkpoint interval:** 5 epochs

---

## Dataset Preparation Notes

1. **Symlink structure:**
   ```
   mmdetection3d/data/kitti/
   ├── training/
   │   ├── calib/
   │   ├── image_2/
   │   ├── label_2/
   │   ├── velodyne/
   │   └── velodyne_reduced -> velodyne
   ├── ImageSets/
   │   ├── train.txt
   │   ├── val.txt
   │   └── test.txt (can be empty)
   ```

2. **Calib files must include:** `Tr_imu_to_velo` transformation matrix
   - **FIXED in carla_dataset_tools:** Now calculates real IMU→LiDAR transformation from actual sensor poses
   - Files modified:
     - `label_tools/kitti_object/kitti_object_data_loader.py` - Added `load_imu_data()` function
     - `label_tools/kitti_objects_label.py` - Loads IMU poses and passes to write_calib
     - `label_tools/kitti_object/kitti_object_helper.py` - Calculates actual `Tr_imu_to_velo` matrix
   - Falls back to identity matrix if IMU data not available

3. **Generate info files:**
   ```bash
   python tools/create_data.py kitti --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti
   ```

---

## Training Command

```bash
python tools/train.py configs/centerpoint/centerpoint_pillar02_second_secfpn_8xb4-cyclic-20e_kitti-3d.py
```

---

## Why These Fixes Were Needed

| Issue | Root Cause |
|-------|-----------|
| `np.long` error | NumPy API deprecation |
| Velocity unpacking | NuScenes vs KITTI format difference (9 vs 7 bbox params) |
| KeyError -1 | KITTI "DontCare" labels not handled |
| Grid size mismatch | SECONDFPN architecture requirement |
| Segfault in eval | Numba JIT compilation issue with KITTI metric |

---

## Files Modified

1. `mmdet3d/datasets/transforms/dbsampler.py`
2. `mmdet3d/models/dense_heads/centerpoint_head.py`
3. `mmdet3d/evaluation/metrics/kitti_metric.py`
4. `configs/centerpoint/centerpoint_pillar02_second_secfpn_8xb4-cyclic-20e_kitti-3d.py` (created)
