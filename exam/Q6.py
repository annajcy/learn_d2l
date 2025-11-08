# Validation Code
import numpy as np

def relu(x):
    output = np.maximum(0, x)
    return output

def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output

def linear_layer_forward(X, weights, bias):
    """
    Forward pass for a linear layer.
    X: input data of shape (n_samples, n_features)
    weights: weights of shape (n_features, n_outputs)
    bias: bias of shape (n_outputs,)
    Returns:
        out: output data of shape (n_samples, n_outputs)
    """
    out = np.dot(X, weights) + bias
    return out

def mlp_forward(X, weights_list, bias_list, activation='sigmoid'):
    """
    Forward pass for a multi-layer perceptron (MLP).
    X: input data of shape (n_samples, n_features)
    weights_list: list of weights for each layer
    bias_list: list of biases for each layer
    activation: activation function to use ('relu' or 'sigmoid')
    Returns:
        out: output data of shape (n_samples, n_outputs)
    """
    
    # select activation function
    if activation == 'relu':
        act_func = relu
    elif activation == 'sigmoid':
        act_func = sigmoid
    else:
        raise ValueError("Activation must be 'relu' or 'sigmoid'")
    
    # initialize current_input
    # this will be updated as we go through layers
    current_input = X
    n_layers = len(weights_list)

    for i in range(n_layers-1):
        current_input = linear_layer_forward(current_input, weights_list[i], bias_list[i])
        current_input = act_func(current_input)

    # forward pass through the last layer (no activation)
    out = linear_layer_forward(current_input, weights_list[-1], bias_list[-1])
    return out

def mlp_backward(X, y, weights_list, bias_list, activation='relu'):
    """
    Backward pass for a multi-layer perceptron (MLP).
    X: input data of shape (n_samples, n_features)
    y: true labels of shape (n_samples, n_outputs)
    weights_list: list of weights for each layer
    bias_list: list of biases for each layer
    activation: activation function to use ('relu' or 'sigmoid')
    Returns: (weight_grads, bias_grads)
        weight_grads: list of gradients w.r.t. weights for each layer
        bias_grads: list of gradients w.r.t. biases for each layer
    """
    n_samples = X.shape[0]
    
    # forward pass to store activations
    if activation == 'relu':
        act_func = relu
    elif activation == 'sigmoid':
        act_func = sigmoid
    else:
        raise ValueError("Activation must be 'relu' or 'sigmoid'")
    
    activations = [X]
    current_input = X
    n_layers = len(weights_list) 
    for i in range(n_layers-1):
        current_input = linear_layer_forward(current_input, weights_list[i], bias_list[i])
        current_input = act_func(current_input)
        activations.append(current_input)
    out = linear_layer_forward(current_input, weights_list[-1], bias_list[-1])
    activations.append(out)
    
    # backward pass to compute gradients
    weight_grads = [None] * n_layers
    bias_grads = [None] * n_layers
    # compute gradient of loss w.r.t. output (mean squared error loss)
    d_out = (out - y) * (2 / n_samples)
    d_current = d_out
    for i in reversed(range(n_layers)):
        a_prev = activations[i]
        # gradients w.r.t. weights and bias
        weight_grads[i] = np.dot(a_prev.T, d_current)
        bias_grads[i] = np.sum(d_current, axis=0)
        
        if i > 0:
            # backpropagate through activation
            d_a_prev = np.dot(d_current, weights_list[i].T)
            if activation == 'relu':
                d_current = d_a_prev * (a_prev > 0)
            elif activation == 'sigmoid':
                sig = a_prev
                d_current = d_a_prev * sig * (1 - sig)
                
    return weight_grads, bias_grads


# Test activation functions
print("Testing activation functions...")
x_test = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
relu_output = relu(x_test)
sigmoid_output = sigmoid(x_test)
print(f"ReLU output: {relu_output}")
print(f"Sigmoid output: {sigmoid_output}")
print("✓ Activation functions test passed!")

# Test linear layer
print("\nTesting linear layer...")
np.random.seed(42)
X_test = np.random.randn(5, 4)
weights_test = np.random.randn(4, 3)
bias_test = np.random.randn(3)
linear_output = linear_layer_forward(X_test, weights_test, bias_test)
print(f"Linear layer output shape: {linear_output.shape}")
assert linear_output.shape == (5, 3), "Linear layer output shape incorrect!"
print("✓ Linear layer test passed!")

# Test MLP forward pass
print("\nTesting MLP forward pass...")
np.random.seed(42)
n_samples, n_features = 10, 4
X = np.random.randn(n_samples, n_features)

# Create a 2-layer MLP: 4 -> 8 -> 1
weights_list = [
    np.random.randn(4, 8) * 0.1,
    np.random.randn(8, 1) * 0.1
]
bias_list = [
    np.zeros(8),
    np.zeros(1)
]

output_relu = mlp_forward(X, weights_list, bias_list, activation='relu')
output_sigmoid = mlp_forward(X, weights_list, bias_list, activation='sigmoid')
print(f"MLP output shape (ReLU): {output_relu.shape}")
print(f"MLP output shape (Sigmoid): {output_sigmoid.shape}")
assert output_relu.shape == (n_samples, 1), "MLP output shape incorrect!"
print("✓ MLP forward pass test passed!")

# Test backward pass
print("\nTesting MLP backward pass...")
y = np.random.randn(n_samples, 1)
w_grads, b_grads = mlp_backward(X, y, weights_list, bias_list, activation='relu')
print(f"Weight gradients shapes: {[g.shape for g in w_grads]}")
print(f"Bias gradients shapes: {[g.shape for g in b_grads]}")
assert len(w_grads) == len(weights_list), "Number of weight gradients doesn't match!"
assert len(b_grads) == len(bias_list), "Number of bias gradients doesn't match!"
print("✓ MLP backward pass test passed!")

