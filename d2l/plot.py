import matplotlib.pyplot as plt
import matplotlib.axes as axes
import numpy as np
from typing import List, Tuple, Optional


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
        
    xlabel, ylabel = label
    xlim, ylim = lim
    xscale, yscale = scale
    for (x, y, fmt) in zip(X_, Y, fmts):
        axes.plot(x, y, fmt) if len(x) else axes.plot(y, fmt)
    set_axes(axes, label, lim, scale, legend)