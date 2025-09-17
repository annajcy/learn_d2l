import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.axes as axes
from matplotlib_inline import backend_inline
from typing import Union, List, Tuple, Any


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
         data: Tuple[List[Any], List[Any]], 
         label: Tuple[str, str], 
         lim: Tuple[Tuple[float, float], Tuple[float, float]], 
         legend: List[str]=[],
         scale: Tuple[str, str]=('linear', 'linear'),
         fmts=('-', 'm--', 'g-.', 'r:'), 
         figsize: Tuple[float, float]=(3.5, 2.5)):

    backend_inline.set_matplotlib_formats('svg')
    plt.rcParams['figure.figsize'] = figsize
    (X, Y) = data
    xlabel, ylabel = label
    xlim, ylim = lim
    xscale, yscale = scale
    for (x, y, fmt) in zip(X, Y, fmts):
        axes.plot(x, y, fmt) if len(x) else axes.plot(y, fmt)
    set_axes(axes, label, lim, scale, legend)