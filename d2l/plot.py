import matplotlib.pyplot as plt
import matplotlib.axes as axes
import numpy as np
from typing import List, Tuple, Optional, Union
import torch

def set_axes(axes: axes.Axes, 
             label: Tuple[str, str], 
             lim: Tuple[Tuple[float, float], Tuple[float, float]], 
             scale: Tuple[str, str], 
             legend: List[str]):
    (xlabel, ylabel) = label
    (xscale, yscale) = scale
    (xlim, ylim) = lim
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()
    
def plot(axes: axes.Axes, 
         data: Tuple[Optional[List[np.ndarray] | np.ndarray], List[np.ndarray]], 
         label: Tuple[str, str], 
         lim: Tuple[Tuple[float, float], Tuple[float, float]], 
         legend: List[str]=[],
         scale: Tuple[str, str]=('linear', 'linear'),
         fmts=('-', 'm--', 'g-.', 'r:'), 
         figsize: Tuple[float, float]=(3.5, 2.5)):

    plt.rcParams['figure.figsize'] = figsize
    (X, Y) = data
    
    if isinstance(X, np.ndarray):
        X_: List[np.ndarray] = [X for i in range(len(Y))]
    elif X is None:
        X_ = [np.arange(len(y)) for y in Y]
    else:
        X_ = X
        
    for (x, y, fmt) in zip(X_, Y, fmts):
        axes.plot(x, y, fmt) if len(x) else axes.plot(y, fmt)
    set_axes(axes, label, lim, scale, legend)
    
def show_images(images: List[Union[np.ndarray, torch.Tensor]], 
                titles: List[str]=[], 
                layout: Tuple[int, int]=(1, 1),
                scale: float=2.0) -> None:
    
    num_rows, num_cols = layout
    figsize = (num_cols * scale, num_rows * scale)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, images)):
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        if img.ndim == 3 and img.shape[0] in (1, 3):  # CHW to HWC
            img = np.transpose(img, (1, 2, 0))
        ax.imshow(img)
        ax.axis('off')
        if titles and i < len(titles):
            ax.set_title(titles[i])
    plt.show()