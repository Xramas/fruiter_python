import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split
import argparse
import json
import os
import time

def main(args):
    """
    Main training function.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Data Processing ---
    IMG_SIZE = 512  # EfficientNetV2-S is often trained on higher resolutions
    
    # Normalization stats for ImageNet
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Transforms for training data (with augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.TrivialAugmentWide(), # Powerful automatic augmentation
        transforms.ToTensor(),
        normalize,
    ])

    # Transforms for validation data (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])

    # --- 2. Dataset and Dataloaders ---
    # Load the full dataset
    full_dataset = datasets.ImageFolder(root=args.data_dir, transform=train_transform)
    
    # Split dataset into training and validation sets
    total_size = len(full_dataset)
    val_size = int(args.val_split * total_size)
    train_size = total_size - val_size
    
    print(f"Dataset found: {total_size} images.")
    print(f"Splitting into {train_size} training and {val_size} validation images.")
    
    # Set different transforms for train and validation datasets
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transform # IMPORTANT: Apply validation transform to the val split

    use_pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=use_pin_memory)
    
    class_names = full_dataset.classes
    num_classes = len(class_names)
    
    # Save class names for GUI
    with open("classes.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=4)
    print(f"Found {num_classes} classes: {class_names}")


    # --- 3. Model Setup ---
    print("Loading pre-trained EfficientNetV2-S model...")
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)

    # Replace the final classifier layer for our number of classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    # --- 4. Loss, Optimizer, and Scheduler ---
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # --- 5. Training Loop ---
    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(args.epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects.double() / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        
        # Update learning rate
        scheduler.step()

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")
            print(f"New best model saved with validation accuracy: {val_acc:.4f}")

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time // 60:.0f}m {total_time % 60:.0f}s")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a fruit classifier with EfficientNetV2-S.')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory for the dataset.')
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size. Reduce if you run out of GPU memory.')
    parser.add_argument('--val_split', type=float, default=0.2, help='Proportion of dataset to use for validation.')
    
    args = parser.parse_args()
    main(args)