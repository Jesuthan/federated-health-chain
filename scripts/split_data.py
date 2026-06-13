#!/usr/bin/env python3
"""
Split Kaggle COVID-19 dataset into 3 hospital folders (non-IID).

Non-IID distribution (reflects real-world heterogeneity):
  Hospital 1 → mostly COVID         (70% COVID,    15% Normal, 15% Viral Pneumonia)
  Hospital 2 → mostly Normal        (15% COVID,    70% Normal, 15% Viral Pneumonia)
  Hospital 3 → mostly Viral Pneum.  (15% COVID,    15% Normal, 70% Viral Pneumonia)

Usage:
  python scripts/split_data.py --data "d:/tmp/COVID-19_Radiography_Dataset"
"""

import argparse
import os
import shutil
import random

# Non-IID split percentages per hospital
# hospital_id → {class_name: fraction}
SPLIT = {
    1: {'COVID': 0.70, 'Normal': 0.15, 'Viral Pneumonia': 0.15},
    2: {'COVID': 0.15, 'Normal': 0.70, 'Viral Pneumonia': 0.15},
    3: {'COVID': 0.15, 'Normal': 0.15, 'Viral Pneumonia': 0.70},
}

CLASSES      = ['COVID', 'Normal', 'Viral Pneumonia']
IMAGES_PER_CLASS = 150   # images per class per hospital (450 total per hospital)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to COVID-19_Radiography_Dataset folder')
    parser.add_argument('--out',  default=None,  help='Output folder (default: fedlearn-fabric/data)')
    parser.add_argument('--per-class', default=150, type=int,
                        help='Images per class per hospital (default: 150)')
    args = parser.parse_args()

    out_dir = args.out or os.path.join(os.path.dirname(__file__), '..', 'data')
    out_dir = os.path.abspath(out_dir)

    random.seed(42)  # reproducible split

    # Collect all images per class
    all_images = {}
    for cls in CLASSES:
        cls_path = os.path.join(args.data, cls, 'images')
        if not os.path.isdir(cls_path):
            cls_path = os.path.join(args.data, cls)
        files = [f for f in os.listdir(cls_path)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(files)
        all_images[cls] = [(f, os.path.join(cls_path, f)) for f in files]
        print(f"  Found {len(files)} images for class: {cls}")

    print(f"\nSplitting into 3 hospitals → {out_dir}\n")

    for hospital_id in [1, 2, 3]:
        fractions = SPLIT[hospital_id]
        print(f"Hospital {hospital_id}  (dominant: {max(fractions, key=fractions.get)})")

        for cls in CLASSES:
            frac    = fractions[cls]
            count   = max(1, int(args.per_class * frac * 3))  # scale by fraction
            count   = min(count, len(all_images[cls]))         # don't exceed available

            dst_dir = os.path.join(out_dir, f'hospital_{hospital_id}', 'covid', cls)
            os.makedirs(dst_dir, exist_ok=True)

            # Pick images — offset by hospital so hospitals get different images
            offset = (hospital_id - 1) * args.per_class
            selected = all_images[cls][offset:offset + count]
            if len(selected) < count:
                selected = all_images[cls][:count]

            for fname, src_path in selected:
                shutil.copy2(src_path, os.path.join(dst_dir, fname))

            print(f"  {cls:<20} → {count} images")

        print()

    # Print summary
    print("=" * 50)
    print("Done! Folder structure created:")
    for hospital_id in [1, 2, 3]:
        total = 0
        for cls in CLASSES:
            p = os.path.join(out_dir, f'hospital_{hospital_id}', 'covid', cls)
            n = len(os.listdir(p)) if os.path.isdir(p) else 0
            total += n
        print(f"  hospital_{hospital_id}/covid/  →  {total} images total")
    print("=" * 50)
    print(f"\nNow run the FL simulation — real data will be used automatically.")


if __name__ == '__main__':
    main()
