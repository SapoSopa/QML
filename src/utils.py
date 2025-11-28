"""
Utility functions for the QML project.
Can be shared across different notebooks.
"""

import numpy as np
import matplotlib.pyplot as plt


def convert_labels_to_quantum(y):
    """
    Convert binary labels {0, 1} to quantum labels {-1, +1}.
    
    Args:
        y: Binary labels (0 or 1)
    
    Returns:
        Quantum labels (-1 or +1)
    """
    return 2 * y - 1


def convert_labels_from_quantum(y_quantum):
    """
    Convert quantum labels {-1, +1} back to binary {0, 1}.
    
    Args:
        y_quantum: Quantum labels (-1 or +1)
    
    Returns:
        Binary labels (0 or 1)
    """
    return (y_quantum + 1) // 2


def plot_dataset(X, y, title="Dataset Visualization"):
    """
    Plot 2D dataset with colored points.
    
    Args:
        X: Input features (N x 2)
        y: Labels
        title: Plot title
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', s=50)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.colorbar(label='Class')
    plt.grid(True, alpha=0.3)
    plt.show()
