import numpy as np
import matplotlib.pyplot as plt
from numpy.ma import indices
from sklearn.datasets import make_blobs
import copy

# Generate 2D data with 4 cluster centers
def generate_data(n_samples=500, std=0.8):
    centers = [(-2, -2), (2, -2), (-2, 2), (2, 2)]
    X, Y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=std, random_state=42)
    return X, Y

# Generate Data
X, Y = generate_data()

# MixUp Augmentation
def mixup(X, Y, alpha=0.8):
    n = X.shape[0]
    lambda_ = np.random.beta(alpha, alpha, n)
    indices = np.random.permutation(n)
    
    # implement mixup
    X_mixup = lambda_[:, np.newaxis] * X + (1 - lambda_)[:, np.newaxis] * X[indices]
    Y_mixup = lambda_ * Y + (1 - lambda_) * Y[indices]
    return X_mixup, Y_mixup

# cutMix Augmentation
def cutmix(X, Y, alpha=0.8):
    n = X.shape[0]
    lambda_ = np.random.beta(alpha, alpha, n)
    indices = np.random.permutation(n)
    
    X_cutmix = copy.deepcopy(X)
    # implement cutmix
    Y_cutmix = lambda_ * Y + (1 - lambda_) * Y[indices]
    
    # compute data range
    x_min, y_min = X.min(axis=0)
    x_max, y_max = X.max(axis=0)
    
    # center point of the box
    center_x = np.random.uniform(x_min, x_max, n)
    center_y = np.random.uniform(y_min, y_max, n)
    
    # lambda sqroot
    lambda_sqrt = np.sqrt(1 - lambda_)
    
    # batch compute the size of the box
    box_w = (x_max - x_min) * lambda_sqrt
    box_h = (y_max - y_min) * lambda_sqrt
    
    # batch compute the box bundaries
    x1, x2 = (center_x - box_w / 2).astype(int), (center_x + box_w / 2).astype(int)
    y1, y2 = (center_y - box_h / 2).astype(int), (center_y + box_h / 2).astype(int)
    
    for i in range(n):
        X_cutmix[i, x1[i]:x2[i]] = X[indices[i], x1[i]:x2[i]]
        Y_cutmix[i] = Y[indices[i]] * (1 - lambda_[i]) + Y[i] * lambda_[i]
        
    return X_cutmix, Y_cutmix

X_mixup, Y_mixup = mixup(X, Y)
X_cutmix, Y_cutmix = cutmix(X, Y)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Raw Data Visualisation
ax[0].scatter(X[:, 0], X[:, 1], c=Y, cmap='coolwarm', alpha=0.8)
ax[0].set_title("Original Data")

# MixUp Visualisation
ax[1].scatter(X_mixup[:, 0], X_mixup[:, 1], c=Y_mixup, cmap='coolwarm', alpha=0.8)
ax[1].set_title("MixUp Data")

# CutMix Visualisation
ax[2].scatter(X_cutmix[:, 0], X_cutmix[:, 1], c=Y_cutmix, cmap='coolwarm', alpha=0.8)
ax[2].set_title("CutMix Data")

plt.show()