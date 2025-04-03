#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RNN implementation for Name Classification
This script implements a basic RNN to classify names by their language/origin
Supports GPU acceleration for training and inference
"""

from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create a results directory if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

# Find all files
def findFiles(path): 
    return glob.glob(path)

# Define the character set
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# Turn a Unicode string to plain ASCII
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []

# Read the data
for filename in findFiles('names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

# Find letter index from all_letters
def letterToIndex(letter):
    return all_letters.find(letter)

# Turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters, device=device)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters> Tensor
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters, device=device)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

# Define the RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        
        # Define the layers
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)  # input to hidden
        self.i2o = nn.Linear(input_size + hidden_size, output_size)  # input to output
        self.softmax = nn.LogSoftmax(dim=1)  # apply log-softmax to output
    
    def forward(self, input, hidden):
        # Combine input and hidden state
        combined = torch.cat((input, hidden), 1)
        
        # Update hidden state
        hidden = self.i2h(combined)
        
        # Calculate output
        output = self.i2o(combined)
        output = self.softmax(output)
        
        return output, hidden
    
    def initHidden(self):
        # Initialize hidden state with zeros
        return torch.zeros(1, self.hidden_size, device=device)

# Function to interpret the output
def categoryFromOutput(output):
    # Get the index of the highest value (most likely category)
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

# Get a random training example
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    # Pick a random category and name
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    
    # Convert category to tensor (index)
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long, device=device)
    
    # Convert name to tensor
    line_tensor = lineToTensor(line)
    
    return category, line, category_tensor, line_tensor

# Time measurement functions
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# Function to train the RNN model
def trainRNN(n_iters=100000, learning_rate=0.005, print_every=5000, plot_every=1000):
    # Create an RNN model
    input_size = n_letters
    hidden_size = 128
    output_size = n_categories
    
    model = RNN(input_size, hidden_size, output_size).to(device)
    
    # Setup optimizer and criterion
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    
    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    
    start = time.time()
    
    for iter in range(1, n_iters + 1):
        # Get random training example
        category, line, category_tensor, line_tensor = randomTrainingExample()
        
        # Initialize hidden state
        hidden = model.initHidden()
        
        # Clear gradients
        model.zero_grad()
        
        # Process each letter in the name
        for i in range(line_tensor.size(0)):
            output, hidden = model(line_tensor[i], hidden)
        
        # Calculate loss
        loss = criterion(output, category_tensor)
        
        # Backpropagate
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        current_loss += loss.item()
        
        # Print progress
        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss.item(), line, guess, correct))
        
        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0
    
    # Save the trained model
    torch.save(model.state_dict(), 'results/rnn_model.pth')
    
    # Plot the training losses
    plt.figure()
    plt.plot(all_losses)
    plt.title('RNN Training Loss')
    plt.xlabel('Iterations (x1000)')
    plt.ylabel('Loss')
    plt.savefig('results/rnn_training_loss.png')
    plt.close()
    
    return all_losses

# Evaluate the model (no gradients needed)
def evaluate(model, line_tensor):
    with torch.no_grad():
        hidden = model.initHidden()
        
        # Process each letter in the name
        for i in range(line_tensor.size(0)):
            output, hidden = model(line_tensor[i], hidden)
        
        return output

# Create confusion matrix for evaluation
def createConfusionMatrix(model, n_confusion=10000):
    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(n_categories, n_categories, device=device)
    
    # Go through a bunch of examples and record which are correctly guessed
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output = evaluate(model, line_tensor)
        guess, guess_i = categoryFromOutput(output)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1
    
    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()
    
    return confusion

# Plot confusion matrix
def plotConfusionMatrix(confusion):
    # Move confusion matrix to CPU for plotting
    confusion_cpu = confusion.cpu()
    
    # Set up plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion_cpu.numpy())
    fig.colorbar(cax)
    
    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)
    
    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    plt.tight_layout()
    plt.savefig('results/rnn_confusion_matrix.png')
    plt.close()

# Predict the language of a name
def predict(model, input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(model, lineToTensor(input_line))
        
        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []
        
        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (math.exp(value), all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])
        
        return predictions

# Calculate accuracy on test set
def calculateAccuracy(model, n_samples=1000):
    correct = 0
    total = 0
    
    for i in range(n_samples):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output = evaluate(model, line_tensor)
        guess, _ = categoryFromOutput(output)
        if guess == category:
            correct += 1
        total += 1
    
    return correct / total

# Main execution
if __name__ == '__main__':
    # Print GPU information if available
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Check if model already exists
    model_path = 'results/rnn_model.pth'
    if os.path.exists(model_path):
        print("Loading pre-trained model...")
        # Load model
        input_size = n_letters
        hidden_size = 128
        output_size = n_categories
        model = RNN(input_size, hidden_size, output_size).to(device)
        
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Will train a new model")
            os.path.exists(model_path) == False  # Force training
    
    if not os.path.exists(model_path):
        print("Training model...")
        # Train model
        losses = trainRNN()
        
        # Load trained model
        input_size = n_letters
        hidden_size = 128
        output_size = n_categories
        model = RNN(input_size, hidden_size, output_size).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Create and plot confusion matrix
    print("Generating confusion matrix...")
    confusion = createConfusionMatrix(model)
    plotConfusionMatrix(confusion)
    
    # Calculate accuracy
    accuracy = calculateAccuracy(model)
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Make some predictions
    test_names = ['Dovesky', 'Jackson', 'Satoshi', 'Hinton', 'Bengio', 'Schmidhuber', 'Wang', 'Hou']
    for name in test_names:
        predict(model, name)
    
    # Print memory usage after all operations
    if torch.cuda.is_available():
        print(f"\nFinal Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Final Memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        
        # Empty cache to free GPU memory
        torch.cuda.empty_cache() 