import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm

# Add src to Python path so we can import modules properly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.model import build_model
from src.data_prep.dataset import get_dataloader

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def main():
    config = load_config()
    root = config['paths']['root']
    
    # Checkpoints and logs paths
    checkpoints_dir = os.path.join(root, config['paths']['checkpoints'])
    logs_dir = os.path.join(root, config['paths']['logs'])
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # DataLoaders
    splits_dir = os.path.join(root, config['data']['splits_dir'])
    train_csv = os.path.join(splits_dir, 'train.csv')
    val_csv = os.path.join(splits_dir, 'val.csv')
    
    print("Loading datasets...")
    train_loader = get_dataloader(train_csv, shuffle=True)
    val_loader = get_dataloader(val_csv, shuffle=False)
    
    # Model
    print("Building model...")
    model = build_model(config).to(device)
    
    # Calculate class weights dynamically to handle the severe data imbalance
    train_df = pd.read_csv(train_csv, header=None, names=['path', 'label'])
    class_counts = train_df['label'].value_counts().sort_index()
    total_samples = len(train_df)
    weights = [total_samples / (2.0 * count) for count in class_counts]
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    
    # Loss and Optimizer with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # Only train parameters that require gradients (LSTM + FC)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['training']['lr'])
    
    epochs = config['training']['epochs']
    best_val_loss = float('inf')
    early_stop_patience = 5
    patience_counter = 0
    
    # Logging
    log_file = os.path.join(logs_dir, 'train_log.csv')
    history = []
    
    print("Starting training...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
        
        # Save logs
        pd.DataFrame(history).to_csv(log_file, index=False)
        
        # Checkpointing & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, 'best_model.pth'))
            print("Saved new best model.")
        else:
            patience_counter += 1
            print(f"No improvement in validation loss. Patience: {patience_counter}/{early_stop_patience}")
            if patience_counter >= early_stop_patience:
                print("Early stopping triggered!")
                break
                
    print("Training finished.")

if __name__ == "__main__":
    main()
