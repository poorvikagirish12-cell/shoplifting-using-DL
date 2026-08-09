import os
import yaml
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class ShopliftingDataset(Dataset):
    def __init__(self, csv_file, config_path="config/config.yaml", transform=None):
        self.config = load_config(config_path)
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
        
        # Resolve paths using config root
        self.root = self.config['paths']['root']
        self.processed_dir = os.path.join(self.root, self.config['data']['processed_dir'])
        
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        rel_path = self.data_frame.iloc[idx, 0]
        label = self.data_frame.iloc[idx, 1]
        
        npy_path = os.path.join(self.processed_dir, rel_path)
        
        # Load numpy array shape: (clip_length, frame_size, frame_size, 3)
        clip = np.load(npy_path)
        
        # Convert to tensor and permute to (C, T, H, W) as expected by many video models
        # like 3D CNNs, or keep as (T, C, H, W) for CNN+LSTM. 
        # We will use (T, C, H, W) because for CNN+LSTM, we process frame by frame.
        clip_tensor = torch.from_numpy(clip).permute(0, 3, 1, 2)
        
        if self.transform:
            clip_tensor = self.transform(clip_tensor)

        return clip_tensor, label

def get_dataloader(csv_file, config_path="config/config.yaml", shuffle=True):
    config = load_config(config_path)
    batch_size = config['training']['batch_size']
    
    dataset = ShopliftingDataset(csv_file=csv_file, config_path=config_path)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return dataloader

if __name__ == "__main__":
    # Test the dataset
    config = load_config()
    splits_dir = os.path.join(config['paths']['root'], config['data']['splits_dir'])
    train_csv = os.path.join(splits_dir, 'train.csv')
    
    if os.path.exists(train_csv):
        ds = ShopliftingDataset(train_csv)
        print(f"Dataset length: {len(ds)}")
        clip, label = ds[0]
        print(f"Clip shape: {clip.shape}, Label: {label}")
    else:
        print(f"{train_csv} not found. Run make_splits.py first.")
