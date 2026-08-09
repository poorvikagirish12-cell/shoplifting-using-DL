import os
import sys
import argparse
import yaml
import torch

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.model import build_model
from src.data_prep.extract_frames import extract_frames_from_video

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def predict_video(video_path, config_path="config/config.yaml"):
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Model
    root = config['paths']['root']
    model_path = os.path.join(root, config['paths']['checkpoints'], 'best_model.pth')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
        
    model = build_model(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 2. Extract and Preprocess Frames
    clip_length = config['data']['clip_length']
    frame_size = config['data']['frame_size']
    
    frames = extract_frames_from_video(video_path, clip_length, frame_size)
    if frames is None:
        raise ValueError(f"Could not extract frames from {video_path}")
        
    # frames is (T, H, W, C) and already normalized to [0, 1]
    # Dataset uses: torch.from_numpy(clip).permute(0, 3, 1, 2) which is (T, C, H, W)
    clip_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    
    # Add batch dimension: (1, T, C, H, W)
    clip_tensor = clip_tensor.unsqueeze(0).to(device)
    
    # 3. Predict
    classes = ['Normal', 'Shoplifting']
    with torch.no_grad():
        outputs = model(clip_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        prob, predicted = torch.max(probabilities, 1)
        
        class_idx = predicted.item()
        confidence = prob.item() * 100
        
    return classes[class_idx], confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Shoplifting from Video")
    parser.add_argument("video_path", type=str, help="Path to raw video file")
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found at {args.video_path}")
        sys.exit(1)
        
    try:
        label, conf = predict_video(args.video_path)
        print("\n" + "="*40)
        print(f"Video: {os.path.basename(args.video_path)}")
        print(f"Prediction: {label}")
        print(f"Confidence: {conf:.2f}%")
        print("="*40 + "\n")
    except Exception as e:
        print(f"Error: {e}")
