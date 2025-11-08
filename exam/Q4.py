import numpy as np
import matplotlib.pyplot as plt

def mae_loss(y_true, y_pred):
    # compute Mean Absolute Error loss, with sign
    return np.mean(y_true - y_pred)

def compute_gradient(X, y, weights, bias):
    n_samples = X.shape[0]
    y_predicted = np.dot(X, weights) + bias
    # compute gradients of mean absolute error loss w.r.t. weights and bias
    error = y_predicted - y
    sign_error = np.sign(error)
    weight_gradient = np.dot(X.T, sign_error) / n_samples
    bias_gradient = np.sum(sign_error) / n_samples
    loss = mae_loss(y, y_predicted)
    return weight_gradient, bias_gradient, loss

def sgd(X_train, y_train, learning_rate=0.01, n_epochs=10, batch_size=16):
    n_samples, n_features = X_train.shape
    
    #define liner regression parameters
    weights = np.zeros(n_features)
    bias = 0.0
    
    for epoch in range(n_epochs):
        indexes = np.arange(n_samples)
        np.random.shuffle(indexes)
        X_train_shuffled = X_train[indexes]
        y_train_shuffled = y_train[indexes]
        
        for i in range(0, n_samples, batch_size):
            # select data batch
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]
            
            weight_gradient, bias_gradient, batch_loss = compute_gradient(X_batch, y_batch, weights, bias)
            
            # update weight and bias
            weights -= learning_rate * weight_gradient
            bias -= learning_rate * bias_gradient
            
    return weights, bias

# Test mae_loss
print("Testing mae_loss...")
y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([1.5, 2.5, 2.5, 4.5, 4.5])
loss = mae_loss(y_true, y_pred)
expected_loss = np.mean(y_true - y_pred)
print(f"MAE Loss: {loss:.4f}, Expected: {expected_loss:.4f}")
assert np.isclose(loss, expected_loss), "mae_loss implementation is incorrect!"
print("✓ mae_loss test passed!")

# Test compute_gradient
print("\nTesting compute_gradient...")
np.random.seed(42)
X_test = np.random.randn(10, 5)
y_test = np.random.randn(10)
weights_test = np.random.randn(5)
bias_test = 0.5

weight_grad, bias_grad, computed_loss = compute_gradient(X_test, y_test, weights_test, bias_test)
print(f"Weight gradient shape: {weight_grad.shape}, Expected shape: ({len(weights_test)},)")
print(f"Bias gradient: {bias_grad:.4f}")
print(f"Computed loss: {computed_loss:.4f}")

assert weight_grad.shape == weights_test.shape, f"Weight gradient shape mismatch"
assert isinstance(bias_grad, (float, np.floating)), f"Bias gradient should be scalar"
print("✓ compute_gradient test passed!")

# Test sgd
print("\nTesting sgd...")
np.random.seed(42)
n_samples, n_features = 200, 10

# Generate synthetic data
true_weights = np.random.randn(n_features)
true_bias = 2.0
X_train = np.random.randn(n_samples, n_features)
y_train = np.dot(X_train, true_weights) + true_bias + 0.1 * np.random.randn(n_samples)

# Run SGD
learned_weights, learned_bias = sgd(X_train, y_train, learning_rate=0.01, n_epochs=50, batch_size=16)

# Verify outputs
assert learned_weights.shape == (n_features,), f"Learned weights shape mismatch"
assert isinstance(learned_bias, (float, np.floating)), f"Learned bias should be scalar"

# Calculate final loss
y_pred_final = np.dot(X_train, learned_weights) + learned_bias
final_loss = mae_loss(y_train, y_pred_final)

print(f"Final MAE Loss: {final_loss:.4f}")
print(f"Learned bias: {learned_bias:.4f}, True bias: {true_bias:.4f}")
print(f"Learned weights shape: {learned_weights.shape}")

# Simple visualization: scatter plot of predictions vs true values
plt.figure(figsize=(8, 6))
plt.scatter(y_train, y_pred_final, alpha=0.6)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2, label='Perfect prediction')
plt.xlabel('True Values', fontsize=12)
plt.ylabel('Predicted Values', fontsize=12)
plt.title('SGD Prediction Results', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

