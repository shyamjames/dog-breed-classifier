import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

def get_dog_classifier(num_classes=120, pretrained=True):
    # Load MobileNetV2 with ImageNet weights
    if pretrained:
        weights = MobileNet_V2_Weights.DEFAULT
        model = mobilenet_v2(weights=weights)
    else:
        model = mobilenet_v2(weights=None)
        
    # We freeze the feature extractor for Phase 1 of training
    for param in model.features.parameters():
        param.requires_grad = False
        
    # Replace the top classifier
    in_features = model.classifier[1].in_features
    # We construct a custom head with Dropout for regularization
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=False),
        nn.Linear(in_features, num_classes)
    )
    
    return model

if __name__ == "__main__":
    # Test the model creation
    model = get_dog_classifier()
    print("Model created.")
    print("Number of trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
