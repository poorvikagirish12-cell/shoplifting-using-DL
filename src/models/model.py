import torch
import torch.nn as nn
import torchvision.models as models

class CNNLSTM(nn.Module):
    def __init__(self, config):
        super(CNNLSTM, self).__init__()
        
        # Load config parameters
        backbone_name = config['model'].get('backbone', 'resnet50')
        hidden_size = config['model'].get('hidden_size', 256)
        num_classes = config['model'].get('num_classes', 2)
        
        # CNN Backbone
        if backbone_name == 'resnet50':
            resnet = models.resnet50(pretrained=True)
            # Remove the classification head (fc layer)
            self.cnn = nn.Sequential(*list(resnet.children())[:-1])
            cnn_out_size = resnet.fc.in_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
            
        # Freeze CNN weights initially to save memory/compute and avoid overfitting
        for param in self.cnn.parameters():
            param.requires_grad = False
            
        # Unfreeze only the last block (layer4) of ResNet50 where task-specific features live
        # layer4 is at index 7 in the sequential children list of ResNet50
        if len(self.cnn) > 7:
            for param in self.cnn[7].parameters():
                param.requires_grad = True
            
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_out_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        # Classification Head
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, clip_length, C, H, W)
        batch_size, clip_length, C, H, W = x.size()
        
        # Reshape to (batch_size * clip_length, C, H, W) to pass through CNN
        x_cnn = x.view(batch_size * clip_length, C, H, W)
        
        # CNN Feature Extraction
        # Output shape: (batch_size * clip_length, cnn_out_size, 1, 1)
        features = self.cnn(x_cnn)
        
        # Flatten and reshape back to (batch_size, clip_length, cnn_out_size)
        features = features.view(batch_size, clip_length, -1)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(features)
        
        # Get the output from the last time step: (batch_size, hidden_size)
        last_out = lstm_out[:, -1, :]
        
        # Classification
        # Output shape: (batch_size, num_classes)
        out = self.fc(last_out)
        
        return out

def build_model(config):
    model_type = config['model'].get('type', 'cnn_lstm')
    
    if model_type == 'cnn_lstm':
        return CNNLSTM(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

if __name__ == "__main__":
    # Unit Test
    print("Running model unit test...")
    dummy_config = {
        'model': {
            'type': 'cnn_lstm',
            'backbone': 'resnet50',
            'hidden_size': 256,
            'num_classes': 2
        }
    }
    
    model = build_model(dummy_config)
    
    # Dummy batch: (batch_size=2, clip_length=16, C=3, H=224, W=224)
    dummy_input = torch.randn(2, 16, 3, 224, 224)
    print(f"Input shape: {dummy_input.shape}")
    
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    
    assert output.shape == (2, 2), f"Expected output shape (2, 2), got {output.shape}"
    print("Unit test passed successfully!")
