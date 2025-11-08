import numpy as np

def conv_forward(X, W, stride=1):
    """
    A naive implementation of the forward pass of a 2D convolutional layer.
    X: input image of shape (H, W)
    W: filter/kernel of shape (HH, WW)
    stride: stride of the convolution
    Returns:
        out: output feature map after convolution
    """
    H, W_in = X.shape
    HH, WW = W.shape

    # Calculate output dimensions
    H_out = 1 + (H - HH) // stride
    W_out = 1 + (W_in - WW) // stride

    out = np.zeros((H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            h_end = h_start + HH
            w_start = j * stride
            w_end = w_start + WW

            out[i, j] = np.sum(X[h_start:h_end, w_start:w_end] * W)

    return out

def conv_backward(d_out, X, W, stride=1):
    """
    A naive implementation of the backward pass of a 2D convolutional layer.
    d_out: gradient of the loss with respect to the output feature map, shape (H_out, W_out)
    X: input image of shape (H, W)
    W: filter/kernel of shape (HH, WW)
    stride: stride of the convolution
    Returns:
        dX: gradient of the loss with respect to the input image X
        dW: gradient of the loss with respect to the filter W
    """
    H, W_in = X.shape
    HH, WW = W.shape
    H_out, W_out = d_out.shape

    dX = np.zeros_like(X)
    dW = np.zeros_like(W)

    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            h_end = h_start + HH
            w_start = j * stride
            w_end = w_start + WW

            # Gradient w.r.t. input X
            dX[h_start:h_end, w_start:w_end] += W * d_out[i, j]

            # Gradient w.r.t. filter W
            dW += X[h_start:h_end, w_start:w_end] * d_out[i, j]

    return dX, dW

# ----- A tiny CNN wrapper (provided) -----
class SimpleCNN:
    def __init__(self, kernel_shape, stride=1, seed=42):
        rng = np.random.default_rng(seed)
        self.kernel = rng.normal(0.0, 0.01, size=kernel_shape)
        self.stride = stride

    def forward(self, X):
        self.X = X
        self.out = conv_forward(X, self.kernel, self.stride)
        return self.out

    def backward(self, d_out):
        dX, dW = conv_backward(d_out, self.X, self.kernel, self.stride)
        self.dW = dW
        return dX

# ----- Numerical gradient checking (provided) -----
def numerical_gradient(cnn, X, d_out, param='kernel', epsilon=1e-5):
    """
    Finite-difference gradient of sum(out * d_out) w.r.t. `param`.
    param in {'kernel', 'input'}.
    """
    param_matrix = cnn.kernel if param == 'kernel' else X
    num_grad = np.zeros_like(param_matrix)
    it = np.nditer(param_matrix, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        original_value = param_matrix[idx]

        param_matrix[idx] = original_value + epsilon
        plus = conv_forward(X if param == 'kernel' else param_matrix, cnn.kernel if param == 'kernel' else cnn.kernel, cnn.stride)
        plus_loss = np.sum(plus * d_out)

        param_matrix[idx] = original_value - epsilon
        minus = conv_forward(X if param == 'kernel' else param_matrix, cnn.kernel if param == 'kernel' else cnn.kernel, cnn.stride)
        minus_loss = np.sum(minus * d_out)

        num_grad[idx] = (plus_loss - minus_loss) / (2 * epsilon)
        param_matrix[idx] = original_value
        it.iternext()
    return num_grad

# ========== (c) Testing script and PRESNT YOUR OUTPUT ==========
if __name__ == "__main__":
    np.random.seed(0)

    X = np.array([
        [ 1,  2,  3,  4,  5],
        [ 6,  7,  8,  9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25]
    ], dtype=float)

    kernel_shape = (3, 3)
    stride = 2

    cnn = SimpleCNN(kernel_shape, stride)
    out = cnn.forward(X)

    d_out = np.array([
        [1, 2],
        [3, 4]
    ], dtype=float)

    # Backward (uses your implementation)
    dX = cnn.backward(d_out)

    # Numerical checks (provided)
    num_grad_kernel = numerical_gradient(cnn, X, d_out, param='kernel')
    num_grad_input  = numerical_gradient(cnn, X, d_out, param='input')

    # Compare (after you implement forward/backward)
    print("Analytical dW:\n", cnn.dW)
    print("Numerical dW:\n", num_grad_kernel)

    rel_err_dw = np.abs(cnn.dW - num_grad_kernel) / (np.maximum(np.abs(cnn.dW), np.abs(num_grad_kernel)) + 1e-8)
    print("Max relative error dW:", np.max(rel_err_dw))

    print("\nAnalytical dX:\n", dX)
    print("Numerical dX:\n", num_grad_input)

    rel_err_dx = np.abs(dX - num_grad_input) / (np.maximum(np.abs(dX), np.abs(num_grad_input)) + 1e-8)
    print("Max relative error dX:", np.max(rel_err_dx))

