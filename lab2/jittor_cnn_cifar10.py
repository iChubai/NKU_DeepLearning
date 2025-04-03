#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CNN for CIFAR10 Image Classification using Jittor
For this tutorial, we will use the CIFAR10 dataset. It has the classes: 
'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'.
The images in CIFAR-10 are of size 3x32x32, i.e. 3-channel color images of 32x32 pixels in size.
"""

import jittor as jt
import jittor.nn as nn
import jittor.transform as transform
import matplotlib.pyplot as plt
import numpy as np
import jittor.optim as optim
import os

# Make deterministic
jt.misc.set_global_seed(42)

# Create results directory
os.makedirs('results', exist_ok=True)

# Functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('results/sample_images_jittor.png')
    plt.close()

# Define CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # 3 input channels, 6 output channels, 5x5 conv kernel
        self.pool = nn.Pool(2, 2, 'maximum')  # 2x2 max pooling
        self.conv2 = nn.Conv2d(6, 16, 5)  # 6 input channels, 16 output channels, 5x5 conv kernel
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 16*5*5 input features, 120 output features
        self.fc2 = nn.Linear(120, 84)  # 120 input features, 84 output features
        self.fc3 = nn.Linear(84, 10)  # 84 input features, 10 output features (for 10 classes)

    def execute(self, x):
        # Apply conv1, then ReLU, then pooling
        x = self.pool(nn.relu(self.conv1(x)))
        # Apply conv2, then ReLU, then pooling
        x = self.pool(nn.relu(self.conv2(x)))
        # Flatten the feature maps
        x = x.view(x.shape[0], -1)
        # Apply fc1, then ReLU
        x = nn.relu(self.fc1(x))
        # Apply fc2, then ReLU
        x = nn.relu(self.fc2(x))
        # Apply fc3
        x = self.fc3(x)
        return x

def main():
    # 1. Load and normalize CIFAR10
    jt.flags.use_cuda = jt.has_cuda  # Use GPU if available
    
    # Define transformations
    transform_train = transform.Compose([
        transform.Resize(32),
        transform.RandomCrop(32, padding=4),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.ImageNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_test = transform.Compose([
        transform.Resize(32),
        transform.ToTensor(),
        transform.ImageNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    batch_size = 4
    
    # Load CIFAR10 dataset
    trainset = jt.dataset.CIFAR10(root='./data', train=True, 
                                 transform=transform_train, download=True)
    trainloader = trainset.set_attrs(batch_size=batch_size, shuffle=True, num_workers=2)
    
    testset = jt.dataset.CIFAR10(root='./data', train=False, 
                                transform=transform_test, download=True)
    testloader = testset.set_attrs(batch_size=batch_size, shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Get some random training images
    for images, labels in trainloader:
        break
    
    # Show images
    print('Sample training images:')
    imshow(jt.misc.make_grid(images))
    print(' '.join(f'{classes[labels[j].item()]:5s}' for j in range(batch_size)))
    
    # 2. Define a Convolutional Neural Network
    net = Net()
    print(f"Using device: {'CUDA' if jt.flags.use_cuda else 'CPU'}")
    
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
            # Get the inputs
            inputs, labels = data
            
            # Forward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            # Zero the parameter gradients + backward + optimize
            optimizer.step(loss)
            
            # Print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # Print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                training_loss.append(running_loss / 2000)
                running_loss = 0.0
    
    print('Finished Training')
    
    # Save the model
    PATH = './results/cifar_net_jittor.pth.tar'
    jt.save(net.state_dict(), PATH)
    print(f'Model saved to {PATH}')
    
    # Plot training loss
    plt.figure(figsize=(10,5))
    plt.plot(training_loss, label='Training Loss')
    plt.xlabel('Iterations (per 2000 batches)')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig('results/training_loss_jittor.png')
    plt.close()
    
    # 5. Test the network on the test data
    # Load the saved model
    net = Net()
    net.load(PATH)
    
    # Get sample test images
    for images, labels in testloader:
        break
    
    # Print the ground truth labels
    print('Ground Truth: ', ' '.join(f'{classes[labels[j].item()]:5s}' for j in range(4)))
    
    # Print the model's predictions
    outputs = net(images)
    predicted = jt.argmax(outputs, 1)[0]
    print('Predicted:    ', ' '.join(f'{classes[predicted[j].item()]:5s}' for j in range(4)))
    
    # Show sample test images
    print('Sample test images:')
    imshow(jt.misc.make_grid(images))
    
    # Evaluate on the entire test set
    correct = 0
    total = 0
    class_correct = {classname: 0 for classname in classes}
    class_total = {classname: 0 for classname in classes}
    
    # No need for with torch.no_grad() in Jittor
    for data in testloader:
        images, labels = data
        # Calculate outputs by running images through the network
        outputs = net(images)
        # The class with the highest energy is what we choose as prediction
        predicted = jt.argmax(outputs, 1)[0]
        total += labels.shape[0]
        correct += (predicted == labels).sum().item()
        
        # Collect accuracy for each class
        for i in range(batch_size):
            if i < len(labels):  # Ensure we don't go out of bounds
                label = labels[i].item()
                class_total[classes[label]] += 1
                if predicted[i].item() == labels[i].item():
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