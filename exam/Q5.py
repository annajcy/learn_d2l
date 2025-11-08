import torchvision, torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def fit_mdae(X_train, p, lambda_reg):
    """
    Uses the closed-form solution to train a marginalized Denoising Autoencoder (mDAE).
    Parameters:
    X_train (np.array): Training data of shape (N, D).
    p (float): Corruption probability (between 0 and 1).
    lambda_reg (float): Regularization parameter (λ). L2 regularization is applied.
    
    Returns:
    W (np.array): Trained weight, shape(D+1, D+1).
    q (np.array): Survival probability vector, shape(D+1,).
    """
    
    X = X_train.T
    D, N = X.shape
    D_plus_1 = D + 1

    # implement fit_mdae using formula WT = (E(Q) + λI)^(-1) E(P) using np.linalg.solve
    # where E(Q) and E(P) are defined in the problem description
    q = np.ones(D_plus_1) * (1 - p)
    q[-1] = 1.0  # Bias term survival probability is 1
    # match S shape (D+1, D+1)
    S = np.zeros((D_plus_1, D_plus_1))
    # Fill S according to the definition
    for i in range(D_plus_1):
        for j in range(D_plus_1):
            if i < D and j < D:
                S[i, j] = (X[i, :] @ X[j, :]) * q[i] * q[j]
            elif i == D and j == D:
                S[i, j] = N
            elif i == D:
                S[i, j] = (X[j, :] @ np.ones(N)) * q[j]
            elif j == D:
                S[i, j] = (X[i, :] @ np.ones(N)) * q[i]
    # E(Q) + λI
    EQ_plus_lambdaI = S + lambda_reg * np.eye(D_plus_1)
    # E(P)
    EP = np.zeros((D_plus_1, D_plus_1))
    for i in range(D_plus_1):
        for j in range(D_plus_1):
            if i < D and j < D:
                EP[i, j] = (X[i, :] @ X[j, :]) * q[j]
            elif i == D and j == D:
                EP[i, j] = N
            elif i == D:
                EP[i, j] = (X[j, :] @ np.ones(N)) * q[j]
            elif j == D:
                EP[i, j] = (X[i, :] @ np.ones(N))
    # Solve for W^T
    W_T = np.linalg.solve(EQ_plus_lambdaI, EP)
    W = W_T.T  # Transpose to get W
    
    print(f"Training completed. W shape: {W.shape}, q shape: {q.shape}")
    return W, q

def reconstruct_mdae(X_clean, W, q):
    """
    Uses the trained W to reconstruct the clean input data.
    
    Parameters:
    X_clean (np.array): Clean input data of shape (N_test, D).
    W (np.array): Trained weight from fit_mdae, shape(D+1, D+1).
    q (np.array): Survival probability vector from fit_mdae, shape(D+1,).
    
    Returns:
    X_recon (np.array): Reconstructed data of shape (N_test, D).
    """
    
    # convert to (D, N_test) format internally
    X = X_clean.T
    D, N_test = X.shape
    
    # implement reconstruct formulas, using W and q
    D_plus_1 = D + 1
    # Add bias term to X
    X_bias = np.vstack([X, np.ones((1, N_test))])
    # Prepare q matrix
    Q_matrix = np.diag(q)
    # Reconstruct
    X_recon_bias = W @ Q_matrix @ X_bias
    # Remove bias term
    X_recon = X_recon_bias[:-1, :].T  # Shape (N_test, D)
    return X_recon

# --- Helper Function (No changes needed) ---
def visualize_results(X_orig, X_recon, p, n_images=10):
    """
    Visualizes original, randomly corrupted, and reconstructed images.
    """
    plt.figure(figsize=(n_images * 2, 4))
    
    # Select random image indices
    indices = np.random.choice(len(X_orig), n_images, replace=False)
    
    for i, idx in enumerate(indices):
        orig_img = X_orig[idx]
        recon_img = X_recon[idx]
        
        # Create one random corruption for visualization
        corruption_mask = np.random.binomial(1, 1-p, size=orig_img.shape)
        corrupted_img = orig_img * corruption_mask
        
        # 1. Original
        ax = plt.subplot(3, n_images, i + 1)
        ax.imshow(orig_img.reshape(28, 28), cmap='gray')
        ax.set_title("Original")
        ax.axis('off')
        
        # 2. Corrupted (for show)
        ax = plt.subplot(3, n_images, i + 1 + n_images)
        ax.imshow(corrupted_img.reshape(28, 28), cmap='gray')
        ax.set_title(f"Corrupted")
        ax.axis('off')
        
        # 3. mDAE Reconstructed
        ax = plt.subplot(3, n_images, i + 1 + 2 * n_images)
        ax.imshow(recon_img.reshape(28, 28), cmap='gray')
        ax.set_title("mDAE Reconstructed")
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

def dataset_to_numpy(dataset, batch_size=256):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    xs = []
    ys = []
    for batch in loader:
        # batch can be (x, y) or just x depending on your dataset
        if isinstance(batch, (list, tuple)):
            x, y = batch
            xs.append(x)
            ys.append(y)
        else:
            x = batch
            xs.append(x)

    X = torch.cat(xs, dim=0).cpu().numpy()
    X = X.reshape(len(X), -1)
    if ys:  # we actually had labels
        y = torch.cat(ys, dim=0).cpu().numpy()
        return X, y
    else:
        return X

mnist = torchvision.datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor()
)

X, y = dataset_to_numpy(mnist, batch_size=512)

def evaluate_results(X_orig, X_recon, p):
    """
    Evaluates reconstruction quality using pixel-level L2 distance.
    
    Parameters:
    X_orig (np.array): Original test data, shape (N, D).
    X_recon (np.array): Reconstructed test data, shape (N, D).
    p (float): Corruption probability used during training.
    """
    # Calculate pixel-level L2 distance for each sample
    l2_distances = np.sqrt(np.sum((X_orig - X_recon) ** 2, axis=1))
    
    # Calculate mean L2 distance
    mean_l2 = np.mean(l2_distances)
    
    print("\n" + "="*60)
    print("Reconstruction Evaluation (Pixel-level L2 Distance)")
    print("="*60)
    print(f"\nMean L2 Distance: {mean_l2:.6f}")
    print(f"Std L2 Distance:  {np.std(l2_distances):.6f}")
    print(f"Min L2 Distance:  {np.min(l2_distances):.6f}")
    print(f"Max L2 Distance:  {np.max(l2_distances):.6f}")
    print("="*60 + "\n")


# --- Main Program (No changes needed) ---
if __name__ == "__main__":
    
    # --- 1. Load and Prepare Data ---
    print("[Main] Loading MNIST data...")
    
    # Normalize pixels from [0, 255] to [0, 1]
    X = X / 255.0
    
    # Use a subset for faster training
    N_TRAIN = 10000
    X_train = X[:N_TRAIN]
    
    # Use a subset for testing
    N_TEST = 100
    X_test = X[N_TRAIN : N_TRAIN + N_TEST]
    
    print(f"[Main] Data loaded. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # --- 2. Set Parameters and Train ---

    CORRUPTION_PROB = 0.25  # (p)
    REGULARIZATION = 1e-5   # (λ)
    
    W_mdae, q_vec = fit_mdae(X_train, p=CORRUPTION_PROB, lambda_reg=REGULARIZATION)
    
    # --- 3. Reconstruct Images ---
    print("[Main] Reconstructing test images...")
    X_reconstructed = reconstruct_mdae(X_test, W_mdae, q_vec)
    
    
    # --- 4. Visualize ---
    print("[Main] Displaying results...")
    if not np.all(X_reconstructed == X_test): # Don't show if functions aren't implemented
        visualize_results(X_test, X_reconstructed, p=CORRUPTION_PROB, n_images=10)
    else:
        print("\n[Main] Functions not implemented. Skipping visualization.")
    
    
    # --- 5. Evaluate Results (Text-based metrics) ---
    print("[Main] Evaluating reconstruction quality...")
    if not np.allclose(X_reconstructed, X_test, atol=1e-6):  # Check if functions are implemented
        # Text-based evaluation
        evaluate_results(X_test, X_reconstructed, p=CORRUPTION_PROB)
    else:
        print("\n[Main] Warning: Reconstruction is identical to input.")
        print("[Main] Functions may not be implemented correctly.")
        print("[Main] Skipping evaluation.")

