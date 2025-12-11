"""
Utility functions for the QML project.
Can be shared across different notebooks.
"""

import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml

# podem ser uteis
def convert_labels_to_quantum(y):
    """
    Convert binary labels {0, 1} to quantum labels {-1, +1}.
    
    Args:
        y: Binary labels (0 or 1)
    
    Returns:
        Quantum labels (-1 or +1)
    """
    return 2 * y - 1


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
def create_variational_circuit(n_qubits, n_layers, diff_method='backprop'):
    """
    Create a variational quantum circuit.
    
    Args:
        n_qubits: Number of qubits
        n_layers: Number of variational layers
        diff_method: Differentiation method (default: 'backprop')
                     - 'backprop': Fast for simulators (automatic differentiation)
                     - 'parameter-shift': Exact, works on real hardware
                     - 'adjoint': Fast and memory-efficient for simulators
                     - 'spsa': Approximation, good for many parameters
    
    Returns:
        QNode: Quantum circuit function
    """
    dev = qml.device('default.qubit', wires=n_qubits)
    
    @qml.qnode(dev, diff_method=diff_method)
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
            
            if n_qubits == 2:
            # Generate CNOT between the two qubits
                qml.CNOT(wires=[1, 0])
            elif n_qubits > 2:
            # Generate CNOTs in a ring topology
                for i in range(n_qubits):
                    qml.CNOT(wires=[(i + 1) % n_qubits, i])
        
        return qml.expval(qml.PauliZ(0))
    
    return circuit


def train_model(optimizer, cost_function, init_params, num_steps, print_interval=10, execs_per_step=None, early_stopping=False, epsilon=0.0001):
    """
    Train a quantum model using a given optimizer.
    
    Args:
        optimizer: PennyLane optimizer instance (e.g., GradientDescentOptimizer, AdamOptimizer, SPSAOptimizer)
        cost_function: Loss function to minimize. Should accept parameters as first argument
        init_params: Initial parameters (will be copied to avoid overwriting)
        num_steps: Number of optimization steps
        print_interval: Print progress every N steps (default: 10)
        execs_per_step: Number of circuit executions per step (optional, for tracking)
                       - None: Estimate based on optimizer type
                       - int: Manual specification
        early_stopping: Enable early stopping when convergence is reached (default: False)
        epsilon: Convergence threshold for early stopping (default: 0.01)
                 Training stops when |cost[k] - cost[k-1]| < epsilon
    
    Returns:
        tuple: (cost_history, exec_history, final_params)
            - cost_history: List of cost values at each step
            - exec_history: List of cumulative circuit executions (if execs_per_step provided)
            - final_params: Optimized parameters
    
    Example:
        >>> opt = qml.AdamOptimizer(stepsize=0.01)
        >>> cost_history, _, params = train_model(opt, loss_fn, params_init, num_steps=100)
    """
    # Copy initial parameters to avoid overwriting (preserve requires_grad)
    params = init_params * 1.0  # Mantém requires_grad=True
    
    # Initialize cost history
    cost_history = []
    initial_cost = cost_function(params)
    cost_history.append(initial_cost)
    
    # Initialize execution history (if tracking)
    exec_history = [0] if execs_per_step is not None else None
    
    # Print header
    optimizer_name = optimizer.__class__.__name__
    print(f"Training with {optimizer_name}")
    print(f"Initial cost: {initial_cost:.6f}")
    print(f"Running {num_steps} optimization steps...")
    if early_stopping:
        print(f"Early stopping enabled (eps={epsilon:.6f})")
    print()
    
    # Training loop
    for step in range(num_steps):
        # Print progress
        if step % print_interval == 0:
            if exec_history is not None:
                print(f"Step {step:4d}/{num_steps} | Executions: {exec_history[step]:6d} | Cost: {cost_history[step]:.6f}")
            else:
                print(f"Step {step:4d}/{num_steps} | Cost: {cost_history[step]:.6f}")
        
        # Perform optimization step
        params = optimizer.step(cost_function, params)
        
        # Monitor cost
        current_cost = cost_function(params)
        cost_history.append(current_cost)
        
        # Track circuit executions (if provided)
        if exec_history is not None:
            exec_history.append((step + 1) * execs_per_step)
        
        # Early stopping check
        if early_stopping and step > 0:
            cost_change = abs(cost_history[-1] - cost_history[-2])
            if cost_change < epsilon:
                print(f"Convergence reached (step {step+1}): |Δcost| = {cost_change:.6f} < {epsilon:.6f}")
                break
    
    # Print final results
    final_cost = cost_history[-1]
    improvement = ((initial_cost - final_cost) / abs(initial_cost)) * 100
    
    print(f"\nFinal cost: {final_cost:.6f}")
    print(f"Improvement: {improvement:+.2f}%")
    if exec_history is not None:
        print(f"Total executions: {exec_history[-1]}")
    print(f"Completed {len(cost_history)-1} steps")
    
    return cost_history, exec_history, params




def train_model_batch(optimizer, loss_function, init_params, X_train, y_train, n_epochs, batch_size, print_interval=10, early_stopping=False, epsilon=0.0001):
    """
    Train a quantum model using mini-batch gradient descent with epochs.
    
    Args:
        optimizer: PennyLane optimizer instance (e.g., GradientDescentOptimizer, AdamOptimizer)
        loss_function: Loss function. Should accept (params, X, y) as arguments
        init_params: Initial parameters (will be copied to avoid overwriting)
        X_train: Training features
        y_train: Training labels (quantum format: -1, +1)
        n_epochs: Number of training epochs
        batch_size: Size of mini-batches for training
        print_interval: Print progress every N epochs (default: 10)
        early_stopping: Enable early stopping when convergence is reached (default: False)
        epsilon: Convergence threshold for early stopping (default: 0.01)
                 Training stops when |loss[k] - loss[k-1]| < epsilon
    
    Returns:
        tuple: (loss_history, final_params)
            - loss_history: List of loss values at each epoch (computed on full training set)
            - final_params: Optimized parameters
    
    Example:
        >>> opt = qml.GradientDescentOptimizer(stepsize=0.1)
        >>> loss_history, params = train_model_batch(
        ...     opt, loss_fn, params_init, X_train, y_train, 
        ...     n_epochs=100, batch_size=10
        ... )
    """
    # Copy initial parameters to avoid overwriting (preserve requires_grad)
    params = init_params * 1.0  # Mantém requires_grad=True
    
    # Calculate initial loss
    initial_loss = loss_function(params, X_train, y_train)
    loss_history = [initial_loss]
    
    # Print header
    optimizer_name = optimizer.__class__.__name__
    print(f"Training with {optimizer_name} (Mini-Batch)")
    print(f"Dataset: {len(X_train)} samples | Batch size: {batch_size}")
    print(f"Initial loss: {initial_loss:.6f}")
    print(f"Running {n_epochs} epochs...")
    if early_stopping:
        print(f"Early stopping enabled (epsilon={epsilon:.6f})")
    print()
    
    # Training loop
    for epoch in range(n_epochs):
        # Shuffle data at each epoch
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        # Mini-batch updates
        for i in range(0, len(X_train), batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Update parameters using mini-batch
            params = optimizer.step(lambda p: loss_function(p, X_batch, y_batch), params)
        
        # Calculate loss on full training set
        current_loss = loss_function(params, X_train, y_train)
        loss_history.append(current_loss)
        
        # Print progress
        if (epoch + 1) % print_interval == 0:
            print(f"Epoch {epoch+1:4d}/{n_epochs} | Loss: {current_loss:.6f}")
        
        # Early stopping check
        if early_stopping and epoch > 0:
            loss_change = abs(loss_history[-1] - loss_history[-2])
            if loss_change < epsilon:
                print(f"Convergence reached (epoch {epoch+1}): |Δloss| = {loss_change:.6f} < {epsilon:.6f}")
                break
    
    # Print final results
    final_loss = loss_history[-1]
    improvement = ((initial_loss - final_loss) / abs(initial_loss)) * 100
    
    print(f"\nTraining Complete!")
    print(f"Final loss: {final_loss:.6f}")
    print(f"Improvement: {improvement:+.2f}%")
    print(f"Completed {len(loss_history)-1} epochs")
    
    return loss_history, params





