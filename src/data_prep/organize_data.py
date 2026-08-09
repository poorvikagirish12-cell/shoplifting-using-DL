import os
import shutil
import pandas as pd
from pathlib import Path
import glob

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Paths
    labels_csv = os.path.join(root_dir, 'temp_dataset', 'DCSASS Dataset', 'Labels', 'Shoplifting.csv')
    source_videos_dir = Path(os.path.join(root_dir, 'temp_dataset', 'DCSASS Dataset', 'Shoplifting'))
    
    raw_shoplifting_dir = os.path.join(root_dir, 'data', 'raw', 'shoplifting')
    raw_normal_dir = os.path.join(root_dir, 'data', 'raw', 'normal')
    
    # Ensure raw directories exist
    os.makedirs(raw_shoplifting_dir, exist_ok=True)
    os.makedirs(raw_normal_dir, exist_ok=True)
    
    # Clean out existing files just to be safe
    print("Cleaning existing raw files...")
    for f in glob.glob(os.path.join(raw_shoplifting_dir, "*.mp4")):
        os.remove(f)
    for f in glob.glob(os.path.join(raw_normal_dir, "*.mp4")):
        os.remove(f)
        
    print("Reading labels...")
    # The CSV has no header, format is: clip_name, class_name, label
    df = pd.read_csv(labels_csv, header=None, names=['clip_name', 'class_name', 'label'])
    
    normal_count = 0
    shoplifting_count = 0
    
    print("Copying files to correct folders...")
    for index, row in df.iterrows():
        clip_name = row['clip_name']
        label = row['label']
        
        # Search for the file recursively since it's nested inside folders
        matches = list(source_videos_dir.rglob(f"{clip_name}.mp4"))
        if not matches:
            print(f"Could not find {clip_name}.mp4")
            continue
            
        source_path = matches[0]
            
        if label == 1:
            target_path = os.path.join(raw_shoplifting_dir, f"{clip_name}.mp4")
            shutil.copy2(str(source_path), target_path)
            shoplifting_count += 1
        elif label == 0:
            target_path = os.path.join(raw_normal_dir, f"{clip_name}.mp4")
            shutil.copy2(str(source_path), target_path)
            normal_count += 1
            
    print(f"Done! Copied {shoplifting_count} Shoplifting videos and {normal_count} Normal videos.")

if __name__ == "__main__":
    main()
