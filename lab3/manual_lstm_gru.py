#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Manual implementation of LSTM and GRU for Name Classification
This script implements LSTM and GRU from scratch without using nn.LSTM or nn.GRU
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

# Basic RNN model definition for comparison
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

# Manual LSTM implementation (from scratch)
class ManualLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ManualLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        
        # Weight matrices and biases for the forget gate
        self.W_f = nn.Parameter(torch.randn(hidden_size, input_size + hidden_size, device=device))
        self.b_f = nn.Parameter(torch.ones(hidden_size, 1, device=device))  # Initialize to 1 to avoid forgetting at beginning
        
        # Weight matrices and biases for the input gate
        self.W_i = nn.Parameter(torch.randn(hidden_size, input_size + hidden_size, device=device))
        self.b_i = nn.Parameter(torch.zeros(hidden_size, 1, device=device))
        
        # Weight matrices and biases for the output gate
        self.W_o = nn.Parameter(torch.randn(hidden_size, input_size + hidden_size, device=device))
        self.b_o = nn.Parameter(torch.zeros(hidden_size, 1, device=device))
        
        # Weight matrices and biases for the cell state
        self.W_c = nn.Parameter(torch.randn(hidden_size, input_size + hidden_size, device=device))
        self.b_c = nn.Parameter(torch.zeros(hidden_size, 1, device=device))
        
        # Output layer
        self.W_out = nn.Parameter(torch.randn(output_size, hidden_size, device=device))
        self.b_out = nn.Parameter(torch.zeros(output_size, 1, device=device))
        
        # Initialize weights with a proper scale
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights with Xavier/Glorot initialization"""
        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            else:
                param.data.fill_(0)
        # Set forget gate bias to 1 as recommended by many LSTM papers
        self.b_f.data.fill_(1)
        
    def forward(self, x, hidden):
        """
        Forward pass for the manual LSTM implementation
        Args:
            x: Input tensor of shape (1, input_size)
            hidden: Tuple of (h, c) where h is hidden state and c is cell state
        Returns:
            output: Output tensor after softmax
            hidden: Tuple of (h, c) updated hidden state and cell state
        """
        h_prev, c_prev = hidden
        
        # Reshape h_prev if needed (from original tensor to needed shape)
        if h_prev.size(0) != 1 or h_prev.size(1) != self.hidden_size:
            h_prev = h_prev.view(1, self.hidden_size)
        if c_prev.size(0) != 1 or c_prev.size(1) != self.hidden_size:
            c_prev = c_prev.view(1, self.hidden_size)
        
        # Combine input and previous hidden state
        combined = torch.cat((x, h_prev), 1)
        combined = combined.t()  # Transpose for matrix multiplication
        
        # Forget gate
        f_gate = torch.sigmoid(self.W_f @ combined + self.b_f)
        
        # Input gate
        i_gate = torch.sigmoid(self.W_i @ combined + self.b_i)
        
        # Output gate
        o_gate = torch.sigmoid(self.W_o @ combined + self.b_o)
        
        # Candidate cell state
        c_candidate = torch.tanh(self.W_c @ combined + self.b_c)
        
        # New cell state
        c_new = f_gate * c_prev.t() + i_gate * c_candidate
        
        # New hidden state
        h_new = o_gate * torch.tanh(c_new)
        
        # Output
        output = self.W_out @ h_new + self.b_out
        
        # Apply log softmax to output
        output = torch.log_softmax(output.t(), dim=1)
        
        return output, (h_new.t(), c_new.t())
    
    def initHidden(self):
        """Initialize hidden state and cell state"""
        return (torch.zeros(1, self.hidden_size, device=device), 
                torch.zeros(1, self.hidden_size, device=device))

# Manual GRU implementation (from scratch)
class ManualGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ManualGRU, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        
        # Weight matrices and biases for the update gate
        self.W_z = nn.Parameter(torch.randn(hidden_size, input_size + hidden_size, device=device))
        self.b_z = nn.Parameter(torch.zeros(hidden_size, 1, device=device))
        
        # Weight matrices and biases for the reset gate
        self.W_r = nn.Parameter(torch.randn(hidden_size, input_size + hidden_size, device=device))
        self.b_r = nn.Parameter(torch.zeros(hidden_size, 1, device=device))
        
        # Weight matrices and biases for the candidate hidden state
        self.W_h = nn.Parameter(torch.randn(hidden_size, input_size + hidden_size, device=device))
        self.b_h = nn.Parameter(torch.zeros(hidden_size, 1, device=device))
        
        # Output layer
        self.W_out = nn.Parameter(torch.randn(output_size, hidden_size, device=device))
        self.b_out = nn.Parameter(torch.zeros(output_size, 1, device=device))
        
        # Initialize weights with a proper scale
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights with Xavier/Glorot initialization"""
        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            else:
                param.data.fill_(0)
        
    def forward(self, x, hidden):
        """
        Forward pass for the manual GRU implementation
        Args:
            x: Input tensor of shape (1, input_size)
            hidden: Previous hidden state tensor
        Returns:
            output: Output tensor after softmax
            hidden: Updated hidden state tensor
        """
        h_prev = hidden
        
        # Reshape h_prev if needed (from original tensor to needed shape)
        if h_prev.size(0) != 1 or h_prev.size(1) != self.hidden_size:
            h_prev = h_prev.view(1, self.hidden_size)
        
        # Combine input and previous hidden state
        combined = torch.cat((x, h_prev), 1)
        combined = combined.t()  # Transpose for matrix multiplication
        
        # Update gate
        z_gate = torch.sigmoid(self.W_z @ combined + self.b_z)
        
        # Reset gate
        r_gate = torch.sigmoid(self.W_r @ combined + self.b_r)
        
        # Candidate hidden state
        # Combine input with reset-gated hidden state
        reset_hidden = r_gate * h_prev.t()
        combined_reset = torch.cat((x.t(), reset_hidden), 0)
        h_candidate = torch.tanh(self.W_h @ combined_reset + self.b_h)
        
        # New hidden state
        h_new = (1 - z_gate) * h_prev.t() + z_gate * h_candidate
        
        # Output
        output = self.W_out @ h_new + self.b_out
        
        # Apply log softmax to output
        output = torch.log_softmax(output.t(), dim=1)
        
        return output, h_new.t()
    
    def initHidden(self):
        """Initialize hidden state"""
        return torch.zeros(1, self.hidden_size, device=device)

# Function to interpret the output
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

# Training function for manual LSTM
def trainManualLSTM(model, n_iters=100000, learning_rate=0.005, print_every=5000, plot_every=1000):
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
    
    return all_losses

# Training function for manual GRU
def trainManualGRU(model, n_iters=100000, learning_rate=0.005, print_every=5000, plot_every=1000):
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
    
    return all_losses

# Evaluate function for manual LSTM and GRU
def evaluateManual(model, line_tensor):
    with torch.no_grad():
        hidden = model.initHidden()
        
        # Process each letter in the name
        for i in range(line_tensor.size(0)):
            output, hidden = model(line_tensor[i], hidden)
        
        return output

# Create confusion matrix
def createConfusionMatrix(model, n_confusion=1000):
    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(n_categories, n_categories, device=device)
    
    # Go through a bunch of examples and record which are correctly guessed
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output = evaluateManual(model, line_tensor)
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

# Prediction function
def predict(model, input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluateManual(model, lineToTensor(input_line))
        
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
        output = evaluateManual(model, line_tensor)
        guess, _ = categoryFromOutput(output)
        if guess == category:
            correct += 1
        total += 1
    
    return correct / total

# Plot the training loss curves
def plotLosses(rnn_losses, lstm_losses, gru_losses, filename='results/manual_loss_comparison.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(rnn_losses, label='RNN')
    plt.plot(lstm_losses, label='Manual LSTM')
    plt.plot(gru_losses, label='Manual GRU')
    plt.title('Training Loss Comparison (Manual Implementation)')
    plt.xlabel('Iterations (x1000)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

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
    lstm_model = ManualLSTM(input_size, hidden_size, output_size).to(device)
    gru_model = ManualGRU(input_size, hidden_size, output_size).to(device)
    
    print(f"Models moved to {device}")
    
    # Try to load the pre-trained RNN model
    try:
        # Load pre-trained model
        state_dict = torch.load('results/rnn_model.pth', map_location=device)
        rnn_model.load_state_dict(state_dict)
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
    
    # Check if models are already trained
    if os.path.exists('results/manual_lstm_model.pth') and os.path.exists('results/manual_gru_model.pth'):
        print("Loading pre-trained models...")
        try:
            # Load LSTM model
            lstm_state_dict = torch.load('results/manual_lstm_model.pth', map_location=device)
            lstm_model.load_state_dict(lstm_state_dict)
            
            # Load GRU model
            gru_state_dict = torch.load('results/manual_gru_model.pth', map_location=device)
            gru_model.load_state_dict(gru_state_dict)
            
            # Load loss curves
            with open('results/manual_lstm_losses.txt', 'r') as f:
                lstm_losses = [float(line.strip()) for line in f.readlines()]
            with open('results/manual_gru_losses.txt', 'r') as f:
                gru_losses = [float(line.strip()) for line in f.readlines()]
            
            print("Pre-trained models loaded successfully")
        except Exception as e:
            print(f"Error loading pre-trained models: {e}")
            print("Will train models from scratch")
            # Continue to training
            os.path.exists('results/manual_lstm_model.pth') == False  # Force training
    
    if not os.path.exists('results/manual_lstm_model.pth') or not os.path.exists('results/manual_gru_model.pth'):
        # Train LSTM model
        print("\nTraining Manual LSTM model...")
        lstm_losses = trainManualLSTM(lstm_model, n_iters=100000)
        torch.save(lstm_model.state_dict(), 'results/manual_lstm_model.pth')
        
        # Save LSTM losses
        with open('results/manual_lstm_losses.txt', 'w') as f:
            for loss in lstm_losses:
                f.write(f"{loss}\n")
        
        # Train GRU model
        print("\nTraining Manual GRU model...")
        gru_losses = trainManualGRU(gru_model, n_iters=100000)
        torch.save(gru_model.state_dict(), 'results/manual_gru_model.pth')
        
        # Save GRU losses
        with open('results/manual_gru_losses.txt', 'w') as f:
            for loss in gru_losses:
                f.write(f"{loss}\n")
    
    # Plot loss comparison
    plotLosses(rnn_losses, lstm_losses, gru_losses)
    
    # Create and plot confusion matrices
    print("\nGenerating confusion matrices...")
    
    lstm_confusion = createConfusionMatrix(lstm_model)
    plotConfusionMatrix(lstm_confusion, "Manual LSTM Confusion Matrix", "results/manual_lstm_confusion_matrix.png")
    
    gru_confusion = createConfusionMatrix(gru_model)
    plotConfusionMatrix(gru_confusion, "Manual GRU Confusion Matrix", "results/manual_gru_confusion_matrix.png")
    
    # Calculate and compare accuracy
    from rnn_name_classifier import evaluate as evaluate_rnn
    
    # Make sure evaluate_rnn function uses the correct device
    def evaluate_rnn_with_device(model, line_tensor):
        line_tensor = line_tensor.to(device)
        with torch.no_grad():
            hidden = model.initHidden()
            for i in range(line_tensor.size(0)):
                output, hidden = model(line_tensor[i], hidden)
            return output
    
    rnn_accuracy = calculateAccuracy(rnn_model)
    lstm_accuracy = calculateAccuracy(lstm_model)
    gru_accuracy = calculateAccuracy(gru_model)
    
    print("\nAccuracy Comparison:")
    print(f"RNN Accuracy: {rnn_accuracy:.4f}")
    print(f"Manual LSTM Accuracy: {lstm_accuracy:.4f}")
    print(f"Manual GRU Accuracy: {gru_accuracy:.4f}")
    
    # Save accuracy results
    with open('results/manual_accuracy_comparison.txt', 'w') as f:
        f.write(f"RNN Accuracy: {rnn_accuracy:.4f}\n")
        f.write(f"Manual LSTM Accuracy: {lstm_accuracy:.4f}\n")
        f.write(f"Manual GRU Accuracy: {gru_accuracy:.4f}\n")
    
    # Run some predictions with each model
    test_names = ['Dovesky', 'Jackson', 'Satoshi', 'Hinton', 'Bengio', 'Schmidhuber', 'Wang', 'Hou']
    
    print("\nRNN Predictions:")
    for name in test_names:
        predict(rnn_model, name)
    
    print("\nManual LSTM Predictions:")
    for name in test_names:
        predict(lstm_model, name)
    
    print("\nManual GRU Predictions:")
    for name in test_names:
        predict(gru_model, name)
    
    # Print memory usage after all operations
    if torch.cuda.is_available():
        print(f"\nFinal Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Final Memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        
        # Empty cache to free GPU memory
        torch.cuda.empty_cache() 