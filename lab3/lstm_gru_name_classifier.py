#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LSTM and GRU implementations for Name Classification
This script implements LSTM and GRU to classify names by their language/origin,
and compares them with the basic RNN model
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

# RNN model definition for comparison
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size, device=device)

# LSTM model definition
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size)
        
        # Output layer
        self.hidden2out = nn.Linear(hidden_size, output_size)
        
        # LogSoftmax for classification
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # input shape: (seq_len, batch, input_size)
        # hidden shape: (1, batch, hidden_size)
        # output shape: (seq_len, batch, hidden_size)
        output, hidden = self.lstm(input, hidden)
        
        # Take the output from the last timestep
        output = self.hidden2out(output[-1])
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        # Returns (h0, c0) for LSTM: initial hidden state and cell state
        return (torch.zeros(1, 1, self.hidden_size, device=device),
                torch.zeros(1, 1, self.hidden_size, device=device))

# GRU model definition
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        
        # GRU layer
        self.gru = nn.GRU(input_size, hidden_size)
        
        # Output layer
        self.hidden2out = nn.Linear(hidden_size, output_size)
        
        # LogSoftmax for classification
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # input shape: (seq_len, batch, input_size)
        # hidden shape: (1, batch, hidden_size)
        # output shape: (seq_len, batch, hidden_size)
        output, hidden = self.gru(input, hidden)
        
        # Take the output from the last timestep
        output = self.hidden2out(output[-1])
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        # Returns initial hidden state for GRU
        return torch.zeros(1, 1, self.hidden_size, device=device)

# Function to interpret the network's output
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

# Get a random training example
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long, device=device)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

# Time measurement functions
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# Training function for LSTM
def trainLSTM(model, n_iters=100000, learning_rate=0.005, print_every=5000, plot_every=1000):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    
    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    
    start = time.time()
    
    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        
        # Initialize hidden state and cell state
        hidden = model.initHidden()
        
        # Clear gradients
        model.zero_grad()
        
        # Reshape line_tensor for LSTM input: (seq_len, batch, input_size)
        reshaped_input = line_tensor.view(line_tensor.size(0), 1, -1)
        
        # Forward pass
        output, hidden = model(reshaped_input, hidden)
        
        # Calculate loss and backpropagate
        loss = criterion(output, category_tensor)
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
    
    return all_losses

# Training function for GRU (similar to LSTM)
def trainGRU(model, n_iters=100000, learning_rate=0.005, print_every=5000, plot_every=1000):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    
    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    
    start = time.time()
    
    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        
        # Initialize hidden state
        hidden = model.initHidden()
        
        # Clear gradients
        model.zero_grad()
        
        # Reshape line_tensor for GRU input: (seq_len, batch, input_size)
        reshaped_input = line_tensor.view(line_tensor.size(0), 1, -1)
        
        # Forward pass
        output, hidden = model(reshaped_input, hidden)
        
        # Calculate loss and backpropagate
        loss = criterion(output, category_tensor)
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
    
    return all_losses

# Evaluation function for LSTM
def evaluateLSTM(model, line_tensor):
    with torch.no_grad():
        hidden = model.initHidden()
        reshaped_input = line_tensor.view(line_tensor.size(0), 1, -1)
        output, _ = model(reshaped_input, hidden)
        return output

# Evaluation function for GRU
def evaluateGRU(model, line_tensor):
    with torch.no_grad():
        hidden = model.initHidden()
        reshaped_input = line_tensor.view(line_tensor.size(0), 1, -1)
        output, _ = model(reshaped_input, hidden)
        return output

# Create confusion matrix
def createConfusionMatrix(model, evaluate_fn, n_confusion=10000):
    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(n_categories, n_categories, device=device)

    # Go through a bunch of examples and record which are correctly guessed
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output = evaluate_fn(model, line_tensor)
        guess, guess_i = categoryFromOutput(output)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()
    
    return confusion

# Plot confusion matrix
def plotConfusionMatrix(confusion, title, filename):
    # Move confusion matrix to CPU for plotting
    confusion_cpu = confusion.cpu()
    
    # Set up plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion_cpu.numpy())
    fig.colorbar(cax)
    
    # Set title
    plt.title(title)

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Plot the training loss curves
def plotLosses(rnn_losses, lstm_losses, gru_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(rnn_losses, label='RNN')
    plt.plot(lstm_losses, label='LSTM')
    plt.plot(gru_losses, label='GRU')
    plt.title('Training Loss Comparison')
    plt.xlabel('Iterations (x1000)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/loss_comparison.png')
    plt.close()

# Prediction function
def predict(model, evaluate_fn, input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate_fn(model, lineToTensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (math.exp(value), all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])
        
        return predictions

# Calculate accuracy on the test set
def calculateAccuracy(model, evaluate_fn, n_samples=1000):
    correct = 0
    total = 0
    
    for i in range(n_samples):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output = evaluate_fn(model, line_tensor)
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
    
    # Set model parameters
    input_size = n_letters
    hidden_size = 128
    output_size = n_categories
    
    # Initialize models
    rnn_model = RNN(input_size, hidden_size, output_size).to(device)
    lstm_model = LSTM(input_size, hidden_size, output_size).to(device)
    gru_model = GRU(input_size, hidden_size, output_size).to(device)
    
    print(f"Models moved to {device}")
    
    # Try to load the pre-trained RNN model
    try:
        rnn_model.load_state_dict(torch.load('results/rnn_model.pth', map_location=device))
        print("Loaded pre-trained RNN model")
        # Assume RNN is already trained, so load its loss curve
        with open('results/rnn_losses.txt', 'r') as f:
            rnn_losses = [float(line.strip()) for line in f.readlines()]
    except Exception as e:
        print(f"Error loading RNN model: {e}")
        print("Training RNN model...")
        # Use standalone RNN training function from rnn_name_classifier.py
        from rnn_name_classifier import trainRNN
        rnn_losses = trainRNN()
        # Save the loss curve for future use
        with open('results/rnn_losses.txt', 'w') as f:
            for loss in rnn_losses:
                f.write(f"{loss}\n")
    
    # Check if LSTM and GRU models are already trained
    lstm_path = 'results/lstm_model.pth'
    gru_path = 'results/gru_model.pth'
    
    if os.path.exists(lstm_path) and os.path.exists(gru_path):
        try:
            lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
            gru_model.load_state_dict(torch.load(gru_path, map_location=device))
            print("Loaded pre-trained LSTM and GRU models")
            
            # Load loss curves
            with open('results/lstm_losses.txt', 'r') as f:
                lstm_losses = [float(line.strip()) for line in f.readlines()]
            with open('results/gru_losses.txt', 'r') as f:
                gru_losses = [float(line.strip()) for line in f.readlines()]
                
        except Exception as e:
            print(f"Error loading LSTM or GRU models: {e}")
            print("Will train models from scratch")
            # Force training
            os.path.exists(lstm_path) == False
    
    # Train LSTM model if needed
    if not os.path.exists(lstm_path):
        print("\nTraining LSTM model...")
        lstm_losses = trainLSTM(lstm_model, n_iters=100000)
        torch.save(lstm_model.state_dict(), lstm_path)
        
        # Save loss curve
        with open('results/lstm_losses.txt', 'w') as f:
            for loss in lstm_losses:
                f.write(f"{loss}\n")
    
    # Train GRU model if needed
    if not os.path.exists(gru_path):
        print("\nTraining GRU model...")
        gru_losses = trainGRU(gru_model, n_iters=100000)
        torch.save(gru_model.state_dict(), gru_path)
        
        # Save loss curve
        with open('results/gru_losses.txt', 'w') as f:
            for loss in gru_losses:
                f.write(f"{loss}\n")
    
    # Plot loss comparison
    plotLosses(rnn_losses, lstm_losses, gru_losses)
    
    # Create and plot confusion matrices
    print("\nGenerating confusion matrices...")
    
    from rnn_name_classifier import evaluate as evaluate_rnn
    
    # Make evaluate_rnn compatible with GPU
    def evaluate_rnn_gpu(model, line_tensor):
        with torch.no_grad():
            hidden = model.initHidden()
            for i in range(line_tensor.size(0)):
                output, hidden = model(line_tensor[i], hidden)
            return output
    
    lstm_confusion = createConfusionMatrix(lstm_model, evaluateLSTM)
    plotConfusionMatrix(lstm_confusion, "LSTM Confusion Matrix", "results/lstm_confusion_matrix.png")
    
    gru_confusion = createConfusionMatrix(gru_model, evaluateGRU)
    plotConfusionMatrix(gru_confusion, "GRU Confusion Matrix", "results/gru_confusion_matrix.png")
    
    # Calculate and compare accuracy
    rnn_accuracy = calculateAccuracy(rnn_model, evaluate_rnn_gpu)
    lstm_accuracy = calculateAccuracy(lstm_model, evaluateLSTM)
    gru_accuracy = calculateAccuracy(gru_model, evaluateGRU)
    
    print("\nAccuracy Comparison:")
    print(f"RNN Accuracy: {rnn_accuracy:.4f}")
    print(f"LSTM Accuracy: {lstm_accuracy:.4f}")
    print(f"GRU Accuracy: {gru_accuracy:.4f}")
    
    # Save accuracy results
    with open('results/accuracy_comparison.txt', 'w') as f:
        f.write(f"RNN Accuracy: {rnn_accuracy:.4f}\n")
        f.write(f"LSTM Accuracy: {lstm_accuracy:.4f}\n")
        f.write(f"GRU Accuracy: {gru_accuracy:.4f}\n")
    
    # Run some predictions with each model
    test_names = ['Dovesky', 'Jackson', 'Satoshi', 'Hinton', 'Bengio', 'Schmidhuber', 'Wang', 'Hou']
    
    print("\nRNN Predictions:")
    for name in test_names:
        predict(rnn_model, evaluate_rnn_gpu, name)
    
    print("\nLSTM Predictions:")
    for name in test_names:
        predict(lstm_model, evaluateLSTM, name)
    
    print("\nGRU Predictions:")
    for name in test_names:
        predict(gru_model, evaluateGRU, name)
        
    # Print memory usage after all operations
    if torch.cuda.is_available():
        print(f"\nFinal Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Final Memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        
        # Empty cache to free GPU memory
        torch.cuda.empty_cache() 