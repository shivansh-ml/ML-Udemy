# Perceptron Learning Algorithm for AND Function
import numpy as np

# Step 1: Prepare input and output (truth table for AND)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Target outputs for AND gate
Y = np.array([0, 0, 0, 1])

# Step 2: Initialize weights and bias
weights = np.array([0.0, 0.0])
bias = 0.0
learning_rate = 0.25

# Activation function (step function)
def step_function(value):
    return 1 if value >= 0 else 0

# Step 3: Training loop
epochs = 10  # maximum iterations
for epoch in range(epochs):
    error_count = 0
    print(f"\nEpoch {epoch+1}")
    for i in range(len(X)):
        # Weighted sum
        summation = np.dot(X[i], weights) + bias
        # Prediction
        y_pred = step_function(summation)
        # Error
        error = Y[i] - y_pred
        if error != 0:
            # Update rule
            weights += learning_rate * error * X[i]
            bias += learning_rate * error
            error_count += 1
        print(f"Input: {X[i]}, Target: {Y[i]}, Predicted: {y_pred}, Weights: {weights}, Bias: {bias}")
    
    # If no error, stop early
    if error_count == 0:
        print("\nTraining converged!")
        break

# Step 4: Final results
print("\nFinal Weights:", weights)
print("Final Bias:", bias)

# Step 5: Testing
print("\nTesting the perceptron:")
for i in range(len(X)):
    summation = np.dot(X[i], weights) + bias
    y_pred = step_function(summation)
    print(f"Input: {X[i]}, Output: {y_pred}")