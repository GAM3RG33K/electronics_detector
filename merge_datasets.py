#!/usr/bin/env python3
"""
YOLO Dataset Merger
Merges multiple YOLO format datasets into a single unified dataset.
Handles class remapping and creates proper train/valid/test splits.
"""

import os
import shutil
import yaml
from pathlib import Path
from collections import OrderedDict
import argparse


class YOLODatasetMerger:
    def __init__(self, source_dir, output_dir):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.datasets = []
        self.unified_classes = []
        self.class_mapping = {}  # {dataset_name: {old_id: new_id}}
        
    def discover_datasets(self):
        """Find all YOLO datasets in the source directory."""
        print("\n🔍 Discovering datasets...")
        for item in self.source_dir.iterdir():
            if item.is_dir() and (item / 'data.yaml').exists():
                self.datasets.append(item)
                print(f"  ✓ Found: {item.name}")
        print(f"\nTotal datasets found: {len(self.datasets)}")
        return self.datasets
    
    def load_dataset_info(self, dataset_path):
        """Load data.yaml from a dataset."""
        yaml_path = dataset_path / 'data.yaml'
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return data
    
    def build_unified_class_list(self):
        """Build a unified list of all classes and create mapping."""
        print("\n📋 Building unified class list...")
        class_set = OrderedDict()
        
        for dataset_path in self.datasets:
            data = self.load_dataset_info(dataset_path)
            dataset_name = dataset_path.name
            classes = data.get('names', [])
            
            print(f"\n  Dataset: {dataset_name}")
            print(f"    Classes ({len(classes)}): {classes}")
            
            # Create mapping for this dataset
            self.class_mapping[dataset_name] = {}
            
            for old_id, class_name in enumerate(classes):
                # Normalize class name (handle duplicates intelligently)
                normalized_name = class_name.strip()
                
                # Check if class already exists (case-insensitive)
                existing_id = None
                for idx, existing_class in enumerate(class_set.keys()):
                    if existing_class.lower() == normalized_name.lower():
                        existing_id = idx
                        break
                
                if existing_id is not None:
                    # Use existing class
                    self.class_mapping[dataset_name][old_id] = existing_id
                    print(f"      {old_id} -> {existing_id}: '{class_name}' (merged with '{list(class_set.keys())[existing_id]}')")
                else:
                    # Add new class
                    new_id = len(class_set)
                    class_set[normalized_name] = new_id
                    self.class_mapping[dataset_name][old_id] = new_id
                    print(f"      {old_id} -> {new_id}: '{class_name}' (new)")
        
        self.unified_classes = list(class_set.keys())
        print(f"\n✅ Unified class list ({len(self.unified_classes)} classes):")
        for idx, cls in enumerate(self.unified_classes):
            print(f"    {idx}: {cls}")
        
        return self.unified_classes
    
    def remap_label_file(self, label_path, dataset_name):
        """Remap class IDs in a YOLO label file."""
        if not label_path.exists():
            return []
        
        remapped_lines = []
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 5:  # class_id x_center y_center width height
                    old_class_id = int(parts[0])
                    new_class_id = self.class_mapping[dataset_name].get(old_class_id, old_class_id)
                    parts[0] = str(new_class_id)
                    remapped_lines.append(' '.join(parts))
        
        return remapped_lines
    
    def copy_and_remap_split(self, dataset_path, split_name, stats):
        """Copy images and remap labels for a specific split (train/valid/test)."""
        dataset_name = dataset_path.name
        source_images = dataset_path / split_name / 'images'
        source_labels = dataset_path / split_name / 'labels'
        
        if not source_images.exists():
            return
        
        dest_images = self.output_dir / split_name / 'images'
        dest_labels = self.output_dir / split_name / 'labels'
        dest_images.mkdir(parents=True, exist_ok=True)
        dest_labels.mkdir(parents=True, exist_ok=True)
        
        # Get all images
        image_files = list(source_images.glob('*.jpg')) + \
                     list(source_images.glob('*.jpeg')) + \
                     list(source_images.glob('*.png'))
        
        for img_path in image_files:
            # Create unique filename to avoid conflicts
            unique_name = f"{dataset_name}_{img_path.name}"
            dest_img_path = dest_images / unique_name
            
            # Copy image
            shutil.copy2(img_path, dest_img_path)
            stats['images'] += 1
            
            # Process corresponding label file
            label_name = img_path.stem + '.txt'
            source_label_path = source_labels / label_name
            dest_label_path = dest_labels / f"{dataset_name}_{label_name}"
            
            if source_label_path.exists():
                remapped_lines = self.remap_label_file(source_label_path, dataset_name)
                if remapped_lines:
                    with open(dest_label_path, 'w') as f:
                        f.write('\n'.join(remapped_lines) + '\n')
                    stats['labels'] += 1
                    stats['annotations'] += len(remapped_lines)
    
    def merge_datasets(self):
        """Merge all datasets into the output directory."""
        print("\n🔄 Merging datasets...")
        
        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        stats = {
            'train': {'images': 0, 'labels': 0, 'annotations': 0},
            'valid': {'images': 0, 'labels': 0, 'annotations': 0},
            'test': {'images': 0, 'labels': 0, 'annotations': 0}
        }
        
        for dataset_path in self.datasets:
            dataset_name = dataset_path.name
            print(f"\n  Processing: {dataset_name}")
            
            for split in ['train', 'valid', 'test']:
                split_path = dataset_path / split
                if split_path.exists():
                    print(f"    - {split}...", end=' ')
                    before = stats[split]['images']
                    self.copy_and_remap_split(dataset_path, split, stats[split])
                    added = stats[split]['images'] - before
                    print(f"✓ ({added} images)")
        
        print("\n📊 Merge Statistics:")
        total_images = 0
        total_annotations = 0
        for split, split_stats in stats.items():
            if split_stats['images'] > 0:
                print(f"  {split.upper()}:")
                print(f"    Images: {split_stats['images']}")
                print(f"    Labels: {split_stats['labels']}")
                print(f"    Annotations: {split_stats['annotations']}")
                total_images += split_stats['images']
                total_annotations += split_stats['annotations']
        
        print(f"\n  TOTAL:")
        print(f"    Images: {total_images}")
        print(f"    Annotations: {total_annotations}")
        print(f"    Classes: {len(self.unified_classes)}")
        
        return stats
    
    def create_data_yaml(self):
        """Create the unified data.yaml file."""
        print("\n📝 Creating data.yaml...")
        
        data_yaml = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'nc': len(self.unified_classes),
            'names': self.unified_classes
        }
        
        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
        
        print(f"  ✓ Created: {yaml_path}")
        return yaml_path
    
    def create_readme(self, stats):
        """Create a README file with dataset information."""
        readme_path = self.output_dir / 'README.md'
        
        with open(readme_path, 'w') as f:
            f.write("# Merged YOLO Dataset\n\n")
            f.write("This dataset was created by merging multiple YOLO format datasets.\n\n")
            
            f.write("## Source Datasets\n\n")
            for dataset in self.datasets:
                f.write(f"- {dataset.name}\n")
            
            f.write("\n## Classes\n\n")
            f.write(f"Total classes: {len(self.unified_classes)}\n\n")
            for idx, cls in enumerate(self.unified_classes):
                f.write(f"{idx}. {cls}\n")
            
            f.write("\n## Dataset Statistics\n\n")
            total_images = 0
            total_annotations = 0
            for split, split_stats in stats.items():
                if split_stats['images'] > 0:
                    f.write(f"### {split.upper()}\n")
                    f.write(f"- Images: {split_stats['images']}\n")
                    f.write(f"- Labels: {split_stats['labels']}\n")
                    f.write(f"- Annotations: {split_stats['annotations']}\n\n")
                    total_images += split_stats['images']
                    total_annotations += split_stats['annotations']
            
            f.write(f"### TOTAL\n")
            f.write(f"- Images: {total_images}\n")
            f.write(f"- Annotations: {total_annotations}\n")
            f.write(f"- Classes: {len(self.unified_classes)}\n")
        
        print(f"  ✓ Created: {readme_path}")
    
    def run(self):
        """Execute the complete merge process."""
        print("="*60)
        print("YOLO Dataset Merger")
        print("="*60)
        
        # Step 1: Discover datasets
        self.discover_datasets()
        
        if not self.datasets:
            print("\n❌ No datasets found!")
            return
        
        # Step 2: Build unified class list
        self.build_unified_class_list()
        
        # Step 3: Merge datasets
        stats = self.merge_datasets()
        
        # Step 4: Create data.yaml
        self.create_data_yaml()
        
        # Step 5: Create README
        self.create_readme(stats)
        
        print("\n" + "="*60)
        print("✅ Merge completed successfully!")
        print(f"📁 Output directory: {self.output_dir.absolute()}")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Merge multiple YOLO format datasets into one unified dataset'
    )
    parser.add_argument(
        '--source',
        type=str,
        default='training-data',
        help='Source directory containing multiple YOLO datasets (default: training-data)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='merged-dataset',
        help='Output directory for merged dataset (default: merged-dataset)'
    )
    
    args = parser.parse_args()
    
    merger = YOLODatasetMerger(args.source, args.output)
    merger.run()


if __name__ == '__main__':
    main()