import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_results(train_losses, val_losses, median_predictions, all_actuals, granger_results, df, model):
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    fig = plt.figure(figsize=(18, 12))

    # Training History
    ax1 = plt.subplot(3, 3, 1)
    epochs_range = range(1, len(train_losses) + 1)
    ax1.plot(epochs_range, train_losses, label='Training Loss', linewidth=2)
    ax1.plot(epochs_range, val_losses, label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training History - TFT Model')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Predictions vs Actual
    ax2 = plt.subplot(3, 3, 2)
    n_samples = min(100, len(median_predictions))
    ax2.plot(all_actuals[:n_samples].numpy(), label='Actual', linewidth=2, marker='o', markersize=3)
    ax2.plot(median_predictions[:n_samples].numpy(), label='Predicted', linewidth=2, marker='s', markersize=3)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Stock Price')
    ax2.set_title('Predictions vs Actual Values')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Additional plots (error, Granger, correlation, etc.) can be added similarly

    plt.tight_layout()
    plt.show()
