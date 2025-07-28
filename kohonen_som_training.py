import numpy as np
import matplotlib.pyplot as plt

# Turn-in 1: Program Code (this entire script)

# Step 1: Generate 1000 points from two circles (input distribution)
def generate_inputs(n=1000):
    theta = np.random.uniform(0, 2*np.pi, n//2)
    circle1 = np.column_stack((1 + 0.5*np.cos(theta), 1 + 0.5*np.sin(theta)))
    circle2 = np.column_stack((3 + 0.5*np.cos(theta), 1 + 0.5*np.sin(theta)))
    return np.vstack((circle1, circle2))

# Step 2: Initialize 100 weights in [2,3] × [0.5,1.5]
def initialize_weights(n=100):
    return np.column_stack((
        np.random.uniform(2, 3, n),
        np.random.uniform(0.5, 1.5, n)
    ))

# Neighborhood function for 1D topology
def get_neighbors(j_star, d, total):
    return [j for j in range(j_star-d, j_star+d+1) if 0 <= j < total]

# Step 3: Training with η₀=2, d₀=3, T=1000
def train_kohonen(inputs, weights, eta0=2.0, d0=3, T=1000):
    n_weights = len(weights)
    snapshots = {0: weights.copy(), 100: None, 1000: None}

    for t in range(1, T+1):
        eta = eta0 * (1 - t/T)
        d = max(1, int(d0 * (1 - t/T)))  # Ensure d ≥ 1
        x = inputs[np.random.randint(len(inputs))]

        # Find winning unit
        distances = np.linalg.norm(weights - x, axis=1)
        j_star = np.argmin(distances)

        # Update weights in neighborhood
        for j in get_neighbors(j_star, d, n_weights):
            weights[j] += eta * (x - weights[j])

        # Store snapshots at required iterations
        if t in snapshots:
            snapshots[t] = weights.copy()

    return snapshots

# Turn-in 2/3: Plotting function for all requirements
def plot_kohonen(inputs, weights, t, turn_in_number):
    plt.figure(figsize=(8, 6))

    # Plot input samples (blue dots)
    plt.scatter(inputs[:,0], inputs[:,1], c='blue', s=10, alpha=0.6,
               label='Input samples (x₁,x₂)')

    # Plot weight vectors (red dots with black edges)
    plt.scatter(weights[:,0], weights[:,1], c='red', s=40,
               edgecolor='black', linewidth=0.5,
               label='Weight vectors (wⱼ₁,wⱼ₂)')

    # Connect immediate neighbors (d=1)
    for j in range(len(weights)):
        neighbors = get_neighbors(j, 1, len(weights))
        for k in neighbors:
            if k > j:  # Avoid duplicate lines
                plt.plot([weights[j,0], weights[k,0]],
                        [weights[j,1], weights[k,1]],
                        'r-', alpha=0.5, linewidth=1)

    # Configure plot based on turn-in number
    if turn_in_number == 2:
        title = "Input Samples and Initial Weights\n" \
                "Blue dots: Input distribution | Red dots: Initial weights\n" \
                "Red lines: Connections between d=1 neighbors"
    else:
        title = f"Weights at t={t}\n" \
                f"Red dots: Weight vectors after {t} iterations\n" \
                "Red lines: Neighborhood connections (d=1)"

    plt.title(title, fontsize=11)
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.xlim(0, 4)
    plt.ylim(0, 2)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()

    # Save plots for PDF submission
    plt.savefig(f"turn_in_{turn_in_number}_{t}.png", dpi=300)
    plt.show()

# Main execution
if __name__ == "__main__":
    # Generate data and initialize weights
    inputs = generate_inputs()
    weights = initialize_weights()

    # Train the network
    snapshots = train_kohonen(inputs, weights.copy())

    # Turn-in 2: Initial state plot
    plot_kohonen(inputs, snapshots[0], 0, 2)

    # Turn-in 3: Evolution plots at t=0, 100, 1000
    plot_kohonen(inputs, snapshots[0], 0, 3)    # Initial state (duplicate of Turn-in 2 but labeled differently)
    plot_kohonen(inputs, snapshots[100], 100, 3)  # Intermediate state
    plot_kohonen(inputs, snapshots[1000], 1000, 3) # Final state
