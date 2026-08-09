import os
import glob
import cv2
import yaml
import numpy as np
from pathlib import Path

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def extract_frames_from_video(video_path, clip_length, frame_size):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < clip_length:
        # If video is shorter than clip_length, we might need to pad or skip
        # For simplicity, we skip if it's extremely short, or duplicate frames
        if frame_count == 0:
            return None
        indices = np.linspace(0, frame_count - 1, clip_length, dtype=int)
    else:
        indices = np.linspace(0, frame_count - 1, clip_length, dtype=int)

    frames = []
    current_idx = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            # Fallback if frame read fails
            if frames:
                frame = frames[-1]
            else:
                frame = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (frame_size, frame_size))
            
        frames.append(frame)
        
    cap.release()
    
    # Shape: (clip_length, frame_size, frame_size, 3)
    frames = np.array(frames)
    
    # Normalize to [0, 1]
    frames = frames.astype(np.float32) / 255.0
    
    # Typically PyTorch expects (C, T, H, W) or (T, C, H, W)
    # We will save as (T, H, W, C) and let the Dataset class handle permutations
    return frames

def main():
    config = load_config()
    raw_dir = config['data']['raw_dir']
    processed_dir = config['data']['processed_dir']
    clip_length = config['data']['clip_length']
    frame_size = config['data']['frame_size']
    
    os.makedirs(processed_dir, exist_ok=True)
    
    classes = ['normal', 'shoplifting']
    
    for cls_name in classes:
        cls_raw_dir = os.path.join(raw_dir, cls_name)
        cls_proc_dir = os.path.join(processed_dir, cls_name)
        os.makedirs(cls_proc_dir, exist_ok=True)
        
        videos = glob.glob(os.path.join(cls_raw_dir, "*.mp4"))
        print(f"Processing {len(videos)} videos for class '{cls_name}'...")
        
        for i, video_path in enumerate(videos):
            vid_name = os.path.basename(video_path)
            out_path = os.path.join(cls_proc_dir, vid_name.replace('.mp4', '.npy'))
            
            # Skip if already processed
            if os.path.exists(out_path):
                continue
                
            frames = extract_frames_from_video(video_path, clip_length, frame_size)
            if frames is not None:
                np.save(out_path, frames)
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(videos)}")

if __name__ == "__main__":
    main()
