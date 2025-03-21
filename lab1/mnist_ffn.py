import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# Set random seed for reproducibility
torch.manual_seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the MLP (Feed-Forward Network)
class MLP(nn.Module):
    """Simple Multi-Layer Perceptron (also called Feed-Forward Network)"""
    def __init__(self, input_dim=784, hidden_dims=[512, 256], output_dim=10, dropout_rate=0.2):
        super().__init__()
        
        # Build layers dynamically based on hidden_dims parameter
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        # Flatten the input image
        x = x.view(x.size(0), -1)
        return self.layers(x)

# Data preparation
def load_data(batch_size=64):
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST datasets
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# Training function
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, target)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = outputs.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    
    return train_loss, train_acc

# Testing function
def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, target)
            
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total
    
    return test_loss, test_acc

# Function to plot training curves
def plot_curves(train_losses, test_losses, train_accs, test_accs, model_name):
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curves')
    plt.legend()
    
    plt.tight_layout()
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{model_name}_curves.png')
    plt.close()

# Main function to run the experiment
def run_experiment(hidden_dims, lr, batch_size, epochs, dropout_rate, model_name):
    print(f"Running experiment with model: {model_name}")
    print(f"Hidden layers: {hidden_dims}")
    print(f"Learning rate: {lr}, Batch size: {batch_size}, Dropout: {dropout_rate}")
    
    # Load data
    train_loader, test_loader = load_data(batch_size)
    
    # Initialize model
    model = MLP(input_dim=28*28, hidden_dims=hidden_dims, output_dim=10, dropout_rate=dropout_rate).to(device)
    print(model)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Lists to store metrics
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    
    start_time = time.time()
    
    # Training loop
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        print(f'Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    
    elapsed_time = time.time() - start_time
    print(f'Training completed in {elapsed_time:.2f} seconds')
    print(f'Best test accuracy: {max(test_accs):.2f}%')
    
    # Plot and save learning curves
    plot_curves(train_losses, test_losses, train_accs, test_accs, model_name)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), f'models/{model_name}.pth')
    
    return model, max(test_accs), train_losses, test_losses, train_accs, test_accs

if __name__ == "__main__":
    # Baseline model configuration
    baseline_config = {
        'hidden_dims': [128, 64],
        'lr': 0.001,
        'batch_size': 64,
        'epochs': 10,
        'dropout_rate': 0.2,
        'model_name': 'baseline'
    }
    
    # Run baseline experiment
    baseline_model, baseline_acc, *_ = run_experiment(**baseline_config) 