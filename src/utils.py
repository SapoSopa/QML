"""
Utility functions for the QML project.
Can be shared across different notebooks.
"""

import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml

# podem ser uteis
# def convert_labels_to_quantum(y):
#     """
#     Convert binary labels {0, 1} to quantum labels {-1, +1}.
    
#     Args:
#         y: Binary labels (0 or 1)
    
#     Returns:
#         Quantum labels (-1 or +1)
#     """
#     return 2 * y - 1


# def convert_labels_from_quantum(y_quantum):
#     """
#     Convert quantum labels {-1, +1} back to binary {0, 1}.
    
#     Args:
#         y_quantum: Quantum labels (-1 or +1)
    
#     Returns:
#         Binary labels (0 or 1)
#     """
#     return (y_quantum + 1) // 2

# Plotting function

def plot_dataset(X, y, title="Dataset Visualization"):
   
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', s=100)
    plt.title(title, fontsize=14)
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.colorbar(label='Class')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'../figures/{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

# Quantum circuit function
@qml.qnode(qml.device('default.qubit', wires=2))

def Angle_embedding(x, n_qubits):

    qml.AngleEmbedding(x, wires=range(n_qubits), rotation='Y')
    return qml.state()


# Função factory para criar o circuito variacional
def create_variational_circuit(n_qubits, n_layers):
    
    dev = qml.device('default.qubit', wires=n_qubits)
    
    @qml.qnode(dev)
    def circuit(params, x):
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation='Y')
        
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.Rot(
                    params[layer, i, 0],  # φ (phi)
                    params[layer, i, 1],  # θ (theta)
                    params[layer, i, 2],  # ω (omega)
                    wires=i
                )
            
            qml.CNOT(wires=[0, 1])
        
        return qml.expval(qml.PauliZ(0))
    
    return circuit