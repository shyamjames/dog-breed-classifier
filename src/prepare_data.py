import os
import shutil
import random
from pathlib import Path

def setup_data_splits(source_dir, dest_dir, train_ratio=0.8, val_ratio=0.1):
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    train_dir = dest_path / 'train'
    val_dir = dest_path / 'val'
    test_dir = dest_path / 'test'
    
    # Create necessary directories
    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Get list of subdirectories (breeds)
    breeds = [d for d in source_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(breeds)} breeds.")
    
    for breed_dir in breeds:
        breed_name = breed_dir.name
        
        # Create breed directories in train/val/test
        (train_dir / breed_name).mkdir(exist_ok=True)
        (val_dir / breed_name).mkdir(exist_ok=True)
        (test_dir / breed_name).mkdir(exist_ok=True)
        
        # Get all images for this breed
        images = list(breed_dir.glob('*.jpg'))
        random.shuffle(images)
        
        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train+n_val]
        test_images = images[n_train+n_val:]
        
        for img in train_images:
            shutil.copy(img, train_dir / breed_name / img.name)
        for img in val_images:
            shutil.copy(img, val_dir / breed_name / img.name)
        for img in test_images:
            shutil.copy(img, test_dir / breed_name / img.name)
            
    print("Data successfully split and organized!")

if __name__ == "__main__":
    SOURCE_DIR = 'data/raw/Images'
    DEST_DIR = 'data/raw'
    setup_data_splits(SOURCE_DIR, DEST_DIR)
