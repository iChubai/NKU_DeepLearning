#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CNN for CIFAR10 Image Classification using PyTorch
For this tutorial, we will use the CIFAR10 dataset. It has the classes: 
'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'.
The images in CIFAR-10 are of size 3x32x32, i.e. 3-channel color images of 32x32 pixels in size.
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

# Make results reproducible
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Create results directory
os.makedirs('results', exist_ok=True)

# Functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('results/sample_images.png')
    plt.close()

def main():
    # 1. Load and normalize CIFAR10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                           shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
             'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # Show images
    print('Sample training images:')
    imshow(torchvision.utils.make_grid(images))
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    # 2. Define a Convolutional Neural Network
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)  # 3 input channels, 6 output channels, 5x5 conv kernel
            self.pool = nn.MaxPool2d(2, 2)   # 2x2 max pooling
            self.conv2 = nn.Conv2d(6, 16, 5) # 6 input channels, 16 output channels, 5x5 conv kernel
            self.fc1 = nn.Linear(16 * 5 * 5, 120) # 16*5*5 input features, 120 output features
            self.fc2 = nn.Linear(120, 84)    # 120 input features, 84 output features
            self.fc3 = nn.Linear(84, 10)     # 84 input features, 10 output features (for 10 classes)

        def forward(self, x):
            # Apply conv1, then ReLU, then pooling
            x = self.pool(F.relu(self.conv1(x)))
            # Apply conv2, then ReLU, then pooling
            x = self.pool(F.relu(self.conv2(x)))
            # Flatten the feature maps
            x = torch.flatten(x, 1)
            # Apply fc1, then ReLU
            x = F.relu(self.fc1(x))
            # Apply fc2, then ReLU
            x = F.relu(self.fc2(x))
            # Apply fc3
            x = self.fc3(x)
            return x

    net = Net()

    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    net.to(device)

    # 3. Define a Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 4. Train the network
    print("Starting training...")
    num_epochs = 2
    training_loss = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # Print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                training_loss.append(running_loss / 2000)
                running_loss = 0.0

    print('Finished Training')

    # Save the model
    PATH = './results/cifar_net.pth'
    torch.save(net.state_dict(), PATH)
    print(f'Model saved to {PATH}')

    # Plot training loss
    plt.figure(figsize=(10,5))
    plt.plot(training_loss, label='Training Loss')
    plt.xlabel('Iterations (per 2000 batches)')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig('results/training_loss.png')
    plt.close()

    # 5. Test the network on the test data
    # Load the saved model
    net = Net()
    net.load_state_dict(torch.load(PATH))
    net.to(device)

    # Get sample test images
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    
    # Print the ground truth labels
    print('Ground Truth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
    
    # Print the model's predictions
    outputs = net(images.to(device))
    _, predicted = torch.max(outputs, 1)
    print('Predicted:    ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))

    # Show sample test images
    print('Sample test images:')
    imshow(torchvision.utils.make_grid(images))

    # Evaluate on the entire test set
    correct = 0
    total = 0
    class_correct = {classname: 0 for classname in classes}
    class_total = {classname: 0 for classname in classes}
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            # Calculate outputs by running images through the network
            outputs = net(images)
            # The class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Collect accuracy for each class
            for i in range(batch_size):
                if i < len(labels):  # Ensure we don't go out of bounds
                    label = labels[i].item()
                    class_total[classes[label]] += 1
                    if predicted[i] == labels[i]:
                        class_correct[classes[label]] += 1

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')

    # Print accuracy for each class
    print('\nAccuracy for each class:')
    for classname in classes:
        if class_total[classname] > 0:
            accuracy = 100 * class_correct[classname] / class_total[classname]
            print(f'Accuracy of {classname:5s} : {accuracy:.2f}%')

if __name__ == '__main__':
    main() 