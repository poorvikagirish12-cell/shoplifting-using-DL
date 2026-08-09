import os
import glob
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    processed_dir = config['data']['processed_dir']
    splits_dir = config['data']['splits_dir']
    os.makedirs(splits_dir, exist_ok=True)
    
    data = []
    classes = {'normal': 0, 'shoplifting': 1}
    
    for cls_name, label in classes.items():
        cls_proc_dir = os.path.join(processed_dir, cls_name)
        npy_files = glob.glob(os.path.join(cls_proc_dir, "*.npy"))
        for f in npy_files:
            # We store relative paths to processed_dir so it works locally and on Colab
            rel_path = os.path.relpath(f, processed_dir)
            # Ensure forward slashes for cross-platform compatibility
            rel_path = rel_path.replace('\\', '/')
            data.append({"clip_path": rel_path, "label": label})
            
    df = pd.DataFrame(data)
    
    if len(df) == 0:
        print("No processed files found. Run extract_frames.py first.")
        return
        
    print(f"Found {len(df)} total clips.")
    
    # 70/20/10 split
    # First split off 10% for test
    train_val, test = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)
    # Then split the remaining 90% into 70/20 (which is ~77.7% / 22.2% of the train_val set)
    train, val = train_test_split(train_val, test_size=(0.2/0.9), stratify=train_val['label'], random_state=42)
    
    train.to_csv(os.path.join(splits_dir, 'train.csv'), index=False)
    val.to_csv(os.path.join(splits_dir, 'val.csv'), index=False)
    test.to_csv(os.path.join(splits_dir, 'test.csv'), index=False)
    
    print(f"Saved splits to {splits_dir}:")
    print(f"  Train: {len(train)}")
    print(f"  Val: {len(val)}")
    print(f"  Test: {len(test)}")

if __name__ == "__main__":
    main()
