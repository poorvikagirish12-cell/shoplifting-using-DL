import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.model import build_model
from src.data_prep.dataset import get_dataloader

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    root = config['paths']['root']
    
    # Paths
    checkpoints_dir = os.path.join(root, config['paths']['checkpoints'])
    results_dir = os.path.join(root, config['paths']['results'])
    os.makedirs(results_dir, exist_ok=True)
    
    model_path = os.path.join(checkpoints_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Please download it from Colab first.")
    
    splits_dir = os.path.join(root, config['data']['splits_dir'])
    test_csv = os.path.join(splits_dir, 'test.csv')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Data
    print("Loading test dataset...")
    test_loader = get_dataloader(test_csv, shuffle=False)
    
    # Load Model
    print("Loading trained model...")
    model = build_model(config)
    # Load weights, map_location handles loading GPU weights to CPU if necessary
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Calculate Metrics
    print("\nGenerating metrics...")
    target_names = ['Normal', 'Shoplifting']
    
    # 1. Classification Report
    report = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0)
    print(report)
    
    metrics_path = os.path.join(results_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("Test Set Classification Report\n")
        f.write("="*40 + "\n")
        f.write(report)
    print(f"Saved metrics to {metrics_path}")
    
    # 2. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix - Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    cm_path = os.path.join(results_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")

if __name__ == "__main__":
    main()
