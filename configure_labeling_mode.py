#!/usr/bin/env python3
"""
Configuration helper for switching between labeling modes

Usage:
    python configure_labeling_mode.py --mode range
    python configure_labeling_mode.py --mode pointcloud
    python configure_labeling_mode.py --status
    python configure_labeling_mode.py --range-size 75.0
"""

import argparse
import sys
from pathlib import Path

HELPER_FILE = Path(__file__).parent / "label_tools/kitti_object/kitti_object_helper.py"


def read_config():
    """Read current configuration from helper file"""
    with open(HELPER_FILE, 'r') as f:
        content = f.read()

    # Extract current settings
    use_range_based = None
    range_box_size = None
    lidar_only_mode = None

    for line in content.split('\n'):
        if 'USE_RANGE_BASED_FILTER' in line and '=' in line:
            use_range_based = 'True' in line
        if 'RANGE_BOX_SIZE' in line and '=' in line:
            parts = line.split('=')
            if len(parts) > 1:
                value_part = parts[1].split('#')[0].strip()
                try:
                    range_box_size = float(value_part)
                except:
                    pass
        if 'LIDAR_ONLY_MODE' in line and '=' in line:
            lidar_only_mode = 'True' in line

    return use_range_based, range_box_size, lidar_only_mode


def write_config(use_range_based=None, range_box_size=None, lidar_only_mode=None):
    """Update configuration in helper file"""
    with open(HELPER_FILE, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if use_range_based is not None and 'USE_RANGE_BASED_FILTER' in line and '=' in line:
            # Update the USE_RANGE_BASED_FILTER line
            indent = len(line) - len(line.lstrip())
            value = "True" if use_range_based else "False"
            spaces = ' ' * indent
            new_line = "{}USE_RANGE_BASED_FILTER = {}  # Set to False to use point cloud-based filtering\n".format(spaces, value)
            new_lines.append(new_line)
        elif range_box_size is not None and 'RANGE_BOX_SIZE' in line and '=' in line:
            # Update the RANGE_BOX_SIZE line
            indent = len(line) - len(line.lstrip())
            spaces = ' ' * indent
            new_line = "{}RANGE_BOX_SIZE = {}  # Detection range in meters (+-{}m in X and Y from sensor)\n".format(spaces, range_box_size, range_box_size)
            new_lines.append(new_line)
        elif lidar_only_mode is not None and 'LIDAR_ONLY_MODE' in line and '=' in line:
            # Update the LIDAR_ONLY_MODE line
            indent = len(line) - len(line.lstrip())
            value = "True" if lidar_only_mode else "False"
            spaces = ' ' * indent
            new_line = "{}LIDAR_ONLY_MODE = {}  # Set to True for 360 LiDAR-only labeling (includes backward objects)\n".format(spaces, value)
            new_lines.append(new_line)
        else:
            new_lines.append(line)

    with open(HELPER_FILE, 'w') as f:
        f.writelines(new_lines)


def show_status():
    """Display current configuration"""
    use_range_based, range_box_size, lidar_only_mode = read_config()

    print("=" * 70)
    print("Current Labeling Configuration")
    print("=" * 70)
    mode_str = 'Range-Based (Stable)' if use_range_based else 'Point Cloud-Based (Traditional)'
    print("Mode: {}".format(mode_str))
    print("Range Box Size: +-{}m".format(range_box_size))
    print("LiDAR-Only Mode: {}".format("ENABLED (360° labeling)" if lidar_only_mode else "DISABLED (camera-based)"))
    print("=" * 70)
    print()

    if lidar_only_mode:
        print("✓ LiDAR-ONLY MODE (360° Coverage)")
        print("  - Labels ALL objects within range box (front + back + sides)")
        print("  - Detection box: [{:.1f}, {:.1f}] x [{:.1f}, {:.1f}] meters".format(
            -range_box_size, range_box_size, -range_box_size, range_box_size))
        print("  - No camera projection filters")
        print("  - 3D labels in LiDAR coordinate system")
    elif use_range_based:
        print("✓ CAMERA-BASED MODE with stable range filtering")
        print("  - Labels objects VISIBLE IN CAMERA (front only)")
        print("  - Detection box: [{:.1f}, {:.1f}] x [{:.1f}, {:.1f}] meters".format(
            -range_box_size, range_box_size, -range_box_size, range_box_size))
        print("  - No point cloud validation required")
        print("  - Based on CARTI_Dataset approach")
    else:
        print("⚠ CAMERA-BASED MODE with point cloud validation")
        print("  - Requires minimum LiDAR points (Cars: 20, Pedestrians: 5)")
        print("  - May miss objects due to occlusion or sensor noise")
        print("  - More realistic but less stable")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Configure labeling mode for CARLA dataset tools',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Switch to stable range-based mode (recommended)
  python configure_labeling_mode.py --mode range

  # Switch to point cloud-based mode
  python configure_labeling_mode.py --mode pointcloud

  # Show current configuration
  python configure_labeling_mode.py --status

  # Change range box size to 75 meters
  python configure_labeling_mode.py --range-size 75.0
        """
    )

    parser.add_argument(
        '--mode', '-m',
        choices=['range', 'pointcloud', 'lidar', 'r', 'p', 'l'],
        help='Labeling mode: range (stable camera), pointcloud (traditional camera), lidar (360 LiDAR-only)'
    )

    parser.add_argument(
        '--range-size', '-r',
        type=float,
        help='Range box size in meters (e.g., 51.2, 75.0, 100.0)'
    )

    parser.add_argument(
        '--status', '-s',
        action='store_true',
        help='Show current configuration'
    )

    args = parser.parse_args()

    # If no arguments, show status
    if not any([args.mode, args.range_size, args.status]):
        show_status()
        return

    # Handle mode change
    if args.mode:
        if args.mode in ['lidar', 'l']:
            # LiDAR-only mode: 360° coverage, no camera filters
            write_config(use_range_based=True, lidar_only_mode=True)
            print("✓ Switched to LiDAR-Only Mode (360° labeling)")
        elif args.mode in ['range', 'r']:
            # Camera-based with stable range filtering
            write_config(use_range_based=True, lidar_only_mode=False)
            print("✓ Switched to Camera-Based Mode with Range Filtering")
        else:
            # Point cloud-based (traditional)
            write_config(use_range_based=False, lidar_only_mode=False)
            print("✓ Switched to Camera-Based Mode with Point Cloud Validation")

    # Handle range size change
    if args.range_size:
        if args.range_size <= 0:
            print("Error: Range size must be positive", file=sys.stderr)
            return 1
        write_config(range_box_size=args.range_size)
        print("✓ Updated range box size to +-{}m".format(args.range_size))

    # Show final status
    print()
    show_status()

    return 0


if __name__ == '__main__':
    sys.exit(main())
