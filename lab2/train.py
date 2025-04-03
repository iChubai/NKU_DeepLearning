# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from models import BasicCNN, MiniResNet, MiniDenseNet, MiniSEResNet
import os
import sys
from tqdm import tqdm

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

BATCH_SIZE = 128
EPOCHS = 20  
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

def get_data_loaders(dataset='cifar10'):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data_path = './data'
    os.makedirs(data_path, exist_ok=True)
    
    print(f"Data will be downloaded to: {os.path.abspath(data_path)}")
    print("If the download is slow, it is recommended to manually download:")
    print("1. CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")
    print("2. Download and save to ./data/cifar-10-python.tar.gz")
    
    if dataset.lower() == 'cifar10':
        try:
            print("Start loading CIFAR-10 dataset...")
            trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                                download=True, transform=transform_train)
            valset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                                download=True, transform=transform_val)
            num_classes = 10
            print("CIFAR-10 dataset loaded!")
        except Exception as e:
            print(f"Failed to download CIFAR-10 dataset: {e}")
            sys.exit(1)
    else:  # cifar100
        try:
            print("Start loading CIFAR-100 dataset...")
            trainset = torchvision.datasets.CIFAR100(root=data_path, train=True,
                                                download=True, transform=transform_train)
            valset = torchvision.datasets.CIFAR100(root=data_path, train=False,
                                                download=True, transform=transform_val)
            num_classes = 100
            print("CIFAR-100 dataset loaded!")
        except Exception as e:
            print(f"Failed to download CIFAR-100 dataset: {e}")
            sys.exit(1)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE,
                                          shuffle=False, num_workers=2)
    
    return trainloader, valloader, num_classes

def train_model(model, trainloader, valloader, model_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0
    
    os.makedirs('results', exist_ok=True)
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{EPOCHS} [Training]')
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            train_pbar.set_postfix({
                'loss': f'{running_loss/len(train_pbar):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
            
        train_loss = running_loss / len(trainloader)
        train_acc = 100. * correct / total
        
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        val_pbar = tqdm(valloader, desc=f'Epoch {epoch+1}/{EPOCHS} [Validation]')
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                val_pbar.set_postfix({
                    'loss': f'{val_loss/len(val_pbar):.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        val_loss = val_loss / len(valloader)
        val_acc = 100. * correct / total
        
        scheduler.step(val_loss)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch+1}/{EPOCHS}:')
        print(f'Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%')
        print(f'Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'results/best_{model_name}.pth')
            print(f"Save the best model, Validation Accuracy: {val_acc:.2f}%")
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Loss Curve')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{model_name} - Accuracy Curve')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'results/{model_name}_curves.png')
    plt.close()
    
    return train_losses, val_losses, train_accs, val_accs

def main():
    os.makedirs('results', exist_ok=True)
    
    trainloader, valloader, num_classes = get_data_loaders('cifar10')
    
    models = {
        'BasicCNN': BasicCNN(),
        'MiniResNet': MiniResNet(num_classes=num_classes),
        'MiniDenseNet': MiniDenseNet(num_classes=num_classes),
        'MiniSEResNet': MiniSEResNet(num_classes=num_classes)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f'\nStart training {name}...')
        print('Model structure:')
        print(model)
        
        model = model.to(DEVICE)
        train_losses, val_losses, train_accs, val_accs = train_model(model, trainloader, valloader, name)
        
        results[name] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        }
    
    plt.figure(figsize=(16, 8))
    
    plt.subplot(1, 2, 1)
    for name, data in results.items():
        plt.plot(data['val_losses'], label=f'{name}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('All models Validation Loss Comparison')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for name, data in results.items():
        plt.plot(data['val_accs'], label=f'{name}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('All models Validation Accuracy Comparison')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/models_comparison.png')
    plt.close()
    
    for name, data in results.items():
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(data['train_losses'], label='Train Loss')
        plt.plot(data['val_losses'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{name} Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(data['train_accs'], label='Train Accuracy')
        plt.plot(data['val_accs'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title(f'{name} Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'results/{name}_training.png')
        plt.close()
    
    print("\nTraining completed. Results saved in 'results' directory.")
if __name__ == '__main__':
    main()