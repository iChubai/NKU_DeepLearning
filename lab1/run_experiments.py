from mnist_ffn import run_experiment
import pandas as pd
import os
import torch

# Set random seed for reproducibility
torch.manual_seed(42)

# Create a directory for results
os.makedirs('results', exist_ok=True)

# Define various experiment configurations
experiments = [
    # Baseline model
    {
        'hidden_dims': [128, 64],
        'lr': 0.001,
        'batch_size': 64,
        'epochs': 15,
        'dropout_rate': 0.2,
        'model_name': 'baseline'
    },
    
    # Experiment with deeper network
    {
        'hidden_dims': [128, 64, 32],
        'lr': 0.001,
        'batch_size': 64,
        'epochs': 15,
        'dropout_rate': 0.2,
        'model_name': 'deeper_network'
    },
    
    # Experiment with wider network
    {
        'hidden_dims': [512, 256],
        'lr': 0.001,
        'batch_size': 64,
        'epochs': 15,
        'dropout_rate': 0.2,
        'model_name': 'wider_network'
    },
    
    # Experiment with learning rate
    {
        'hidden_dims': [128, 64],
        'lr': 0.01,
        'batch_size': 64,
        'epochs': 15,
        'dropout_rate': 0.2,
        'model_name': 'higher_lr'
    },
    
    # Experiment with different batch size
    {
        'hidden_dims': [128, 64],
        'lr': 0.001,
        'batch_size': 128,
        'epochs': 15,
        'dropout_rate': 0.2,
        'model_name': 'larger_batch'
    },
    
    # Experiment with dropout
    {
        'hidden_dims': [128, 64],
        'lr': 0.001,
        'batch_size': 64,
        'epochs': 15,
        'dropout_rate': 0.5,
        'model_name': 'higher_dropout'
    },
    
    # Experiment with no dropout
    {
        'hidden_dims': [128, 64],
        'lr': 0.001,
        'batch_size': 64,
        'epochs': 15,
        'dropout_rate': 0.0,
        'model_name': 'no_dropout'
    },
    
    # Best model based on previous experiments (to be modified after analyzing results)
    {
        'hidden_dims': [512, 256, 128],
        'lr': 0.001,
        'batch_size': 128,
        'epochs': 20,
        'dropout_rate': 0.3,
        'model_name': 'best_model'
    }
]

# Run all experiments and collect results
results = []
for config in experiments:
    print(f"\n{'='*50}")
    print(f"Running experiment: {config['model_name']}")
    print(f"{'='*50}")
    
    model, best_acc, train_losses, test_losses, train_accs, test_accs = run_experiment(**config)
    
    # Store results
    results.append({
        'model_name': config['model_name'],
        'hidden_dims': str(config['hidden_dims']),
        'learning_rate': config['lr'],
        'batch_size': config['batch_size'],
        'dropout_rate': config['dropout_rate'],
        'best_accuracy': best_acc,
        'final_train_loss': train_losses[-1],
        'final_test_loss': test_losses[-1]
    })

# Create results dataframe and save to CSV
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('best_accuracy', ascending=False)
results_df.to_csv('results/experiment_results.csv', index=False)

print("\nAll experiments completed!")
print("Top 3 configurations:")
print(results_df.head(3).to_string(index=False)) 