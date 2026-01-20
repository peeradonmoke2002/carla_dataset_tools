# Quick Start Guide - CARLA Data Collection

## ✅ What's Already Configured

1. **✓ Pedestrian Filtering**: Only normal adult pedestrians (no children, police, or wheelchair users)
2. **✓ Range-Based Labeling**: Stable labeling enabled (±51.2m detection box)

## 🚀 How to Start Data Collection

### Step 1: Start CARLA Server

```bash
# In one terminal, start CARLA
cd /path/to/CARLA_0.9.XX
./CarlaUE4.sh

# Or for headless mode (faster):
./CarlaUE4.sh -RenderOffScreen
```

### Step 2: Run Data Recorder

```bash
# Navigate to the project directory
cd /home/peeradon/autoware/src/av-stack-playground/trainCenterPoint_carla/carla_dataset_tools

# Run with default config
python data_recorder.py --host 127.0.0.1 --port 2000

# Or with specific profile
python data_recorder.py --profile default

# Or with custom config
python data_recorder.py --config /path/to/your/config.yaml
```

### Available Profiles

Check available profiles:
```bash
ls config/profiles/
```

Common profiles:
- `default` - Standard configuration
- `kitti` - KITTI-like dataset
- `simple` - Simple test configuration

## 📊 Generated Data Structure

Data will be saved to:
```
raw_data/record_YYYY_MMDD_HHMM/
├── others.world_0/              # Ground truth labels (pickle files)
├── vehicle.tesla.model3_1/      # Ego vehicle data
│   ├── velodyne/                # LiDAR point clouds (.ply)
│   ├── image_2/                 # Camera images (.png)
│   ├── camera_info.json         # Camera calibration
│   └── lidar_info.json          # LiDAR metadata
└── carla_raw_record.log         # CARLA replay log
```

## 🏷️ Generate Labels (After Data Collection)

### Step 1: Generate KITTI Format Labels

```bash
cd /home/peeradon/autoware/src/av-stack-playground/trainCenterPoint_carla/carla_dataset_tools

# For all vehicles in a recording
python label_tools/kitti_objects_label.py \
    --record record_2024_0120_1234 \
    --vehicle all

# For specific vehicle
python label_tools/kitti_objects_label.py \
    --record record_2024_0120_1234 \
    --vehicle vehicle.tesla.model3_1 \
    --lidar velodyne \
    --camera image_2
```

### Output Structure

Labels will be generated in:
```
dataset/record_YYYY_MMDD_HHMM/vehicle.tesla.model3_1/kitti_object/
├── training/
│   ├── calib/           # Calibration files
│   ├── image_2/         # RGB images
│   ├── label_2/         # KITTI format labels
│   └── velodyne/        # Point clouds (.bin)
└── ImageSets/
    ├── train.txt
    └── val.txt
```

## ⚙️ Configuration Options

### Check Current Labeling Mode
```bash
python3 configure_labeling_mode.py --status
```

### Change Detection Range
```bash
# Set to 75 meters
python3 configure_labeling_mode.py --range-size 75.0

# Set to 100 meters
python3 configure_labeling_mode.py --range-size 100.0
```

### Switch to Point Cloud Mode (if needed)
```bash
python3 configure_labeling_mode.py --mode pointcloud
```

### Switch Back to Range-Based (Recommended)
```bash
python3 configure_labeling_mode.py --mode range
```

## 📝 Example: Complete Workflow

```bash
# Terminal 1: Start CARLA
cd /path/to/CARLA_0.9.XX
./CarlaUE4.sh

# Terminal 2: Collect data
cd /home/peeradon/autoware/src/av-stack-playground/trainCenterPoint_carla/carla_dataset_tools
python data_recorder.py --profile default

# Wait for recording to complete (Press Ctrl+C to stop early)

# Generate labels (get record name from output above)
python label_tools/kitti_objects_label.py \
    --record record_2024_0120_1530 \
    --vehicle all
```

## 🔧 Troubleshooting

### "Connection refused" Error
- Make sure CARLA server is running
- Check port (default: 2000)
- Try: `python data_recorder.py --host 127.0.0.1 --port 2000`

### "Too many/few labels"
- Adjust detection range:
  ```bash
  python3 configure_labeling_mode.py --range-size 30.0  # Smaller range
  python3 configure_labeling_mode.py --range-size 75.0  # Larger range
  ```

### "Missing pedestrians in labels"
- Make sure you're using range-based mode:
  ```bash
  python3 configure_labeling_mode.py --mode range
  ```

### "Only want adult pedestrians"
- Already configured! The system filters out:
  - Children (walker.pedestrian.0009-0014, 0048, 0049)
  - Police (walker.pedestrian.0030, 0032)
  - Wheelchair users (can_use_wheelchair attribute)

## 📖 More Information

- [README_LABELING_FIX.md](../README_LABELING_FIX.md) - Labeling stability improvements
- [RANGE_BASED_LABELING.md](../RANGE_BASED_LABELING.md) - Technical details
- [Config documentation](config/README.md) - Configuration file format

## 🎯 Key Features Enabled

✅ **Stable Range-Based Labeling** (no point cloud dependency)
✅ **Normal Adult Pedestrians Only** (no children/police/wheelchair)
✅ **Configurable Detection Range** (default: ±51.2m)
✅ **KITTI Format Output** (compatible with 3D detection models)
✅ **Multi-Vehicle Support** (collect from multiple vehicles)
