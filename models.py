"""
Model architectures for rice disease classification
"""
import torch
import torch.nn as nn
import timm

class RiceClassifier(nn.Module):
    def __init__(self, model_name='efficientnet_b0', num_classes=20, pretrained=True):
        super(RiceClassifier, self).__init__()
        
        # Load pre-trained model
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # Get number of features
        num_features = self.backbone.num_features
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output


def create_model(model_name='efficientnet_b0', num_classes=20):
    """
    Create model from various architectures
    
    Recommended models:
    - efficientnet_b0, efficientnet_b3 (balanced)
    - resnet50, resnet101 (robust)
    - vit_base_patch16_224 (transformer-based)
    - convnext_base (modern CNN)
    """
    model = RiceClassifier(model_name, num_classes)
    return model


if __name__ == "__main__":
    # Test model creation
    model = create_model('efficientnet_b0', num_classes=20)
    print(model)
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"\nOutput shape: {output.shape}")