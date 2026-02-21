import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import get_dog_classifier
import copy

def get_dataloaders(data_dir, batch_size=32):
    # ImageNet normalization stats
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]),
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val']
    }
    
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0)
        for x in ['train', 'val']
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=10, device='cpu'):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for bi, (inputs, labels) in enumerate(dataloaders[phase]):
                # if bi >= 2: # Limit to 2 batches for fast dev test
                #     break
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'models/best_model.pth')

    print(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    data_dir = 'data/raw'
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataloaders, dataset_sizes, class_names = get_dataloaders(data_dir, batch_size=32)
    
    # PHASE 1: Train just the head
    print("===== PHASE 1: Training Custom Head =====")
    model = get_dog_classifier(num_classes=120, pretrained=True)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    # Optimize only the classifier parameters
    optimizer_ft = optim.Adam(model.classifier.parameters(), lr=0.001)
    
    # Train for a few epochs as a test / Phase 1
    # Note: given CPU is likely being used, we'll keep epochs low for demonstration
    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer_ft, num_epochs=1, device=device)
    
    # PHASE 2: Unfreeze base layers and fine-tune
    print("===== PHASE 2: Fine-Tuning =====")
    for param in model.features.parameters():
        param.requires_grad = True
        
    optimizer_fine = optim.Adam(model.parameters(), lr=1e-5)
    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer_fine, num_epochs=1, device=device)
    
    print("Training completely finished.")
