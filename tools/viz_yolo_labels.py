#!/usr/bin/env python3
"""Visualize YOLO labels on images to verify correctness."""
import cv2
import os
import sys
import glob
import random

# Class names matching COCO indices used in yolov5_helper.py
CLASS_NAMES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
    9: 'traffic light',
    80: 'traffic sign',
    81: 'TL green',
    82: 'TL red',
}

# Colors for each class (BGR)
CLASS_COLORS = {
    0: (60, 20, 220),    # person - red
    1: (32, 11, 119),    # bicycle - dark red
    2: (142, 0, 0),      # car - blue
    3: (230, 0, 0),      # motorcycle - blue
    5: (100, 60, 0),     # bus - dark blue
    7: (70, 0, 0),       # truck - dark blue
    9: (30, 170, 250),   # traffic light - orange
    80: (0, 220, 220),   # traffic sign - yellow
    81: (0, 255, 0),     # TL green - green
    82: (0, 0, 255),     # TL red - red
}

def draw_yolo_labels(image_path, label_path):
    """Draw YOLO bounding boxes on image."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return None
    
    h, w = img.shape[:2]
    
    if not os.path.exists(label_path):
        print(f"No label file: {label_path}")
        return img
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            x_center = float(parts[1]) * w
            y_center = float(parts[2]) * h
            box_w = float(parts[3]) * w
            box_h = float(parts[4]) * h
            
            x1 = int(x_center - box_w / 2)
            y1 = int(y_center - box_h / 2)
            x2 = int(x_center + box_w / 2)
            y2 = int(y_center + box_h / 2)
            
            color = CLASS_COLORS.get(class_id, (255, 255, 255))
            label = CLASS_NAMES.get(class_id, f'class_{class_id}')
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img

def main():
    if len(sys.argv) < 2:
        print("Usage: python viz_yolo_labels.py <yolo_dataset_path> [num_samples]")
        print("Example: python viz_yolo_labels.py dataset/record_xxx/vehicle_1st/yolo 5")
        sys.exit(1)
    
    yolo_path = sys.argv[1]
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    images_dir = os.path.join(yolo_path, 'yolo_dataset', 'images', 'train')
    labels_dir = os.path.join(yolo_path, 'yolo_dataset', 'labels', 'train')
    
    if not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        sys.exit(1)
    
    image_files = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))
    if not image_files:
        print("No images found")
        sys.exit(1)
    
    # Sample random images
    samples = random.sample(image_files, min(num_samples, len(image_files)))
    
    output_dir = os.path.join(yolo_path, 'viz_samples')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Visualizing {len(samples)} samples...")
    for img_path in samples:
        basename = os.path.basename(img_path).replace('.jpg', '')
        label_path = os.path.join(labels_dir, basename + '.txt')
        
        result = draw_yolo_labels(img_path, label_path)
        if result is not None:
            output_path = os.path.join(output_dir, f'viz_{basename}.jpg')
            cv2.imwrite(output_path, result)
            print(f"  Saved: {output_path}")
    
    print(f"\nDone! Check visualizations in: {output_dir}")

if __name__ == '__main__':
    main()
