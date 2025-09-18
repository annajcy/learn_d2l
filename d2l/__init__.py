from .plot import (
    plot, 
    set_axes,
    show_images
)

from .optimizer import (
    SGD
)

from .regression import (
    SyntheticRegressionDataScratch, 
    SyntheticRegressionDataTorch, 
    LinearRegressionScratch, 
    LinearRegressionTorch, 
    LinearRegressionTorchL2
)

from .classification import (
    FashionMNIST,
    SoftmaxClassifierScratch
)

__all__ = [
    # Plot utilities
    'plot', 
    'set_axes',
    'show_images',
    
    # Optimizer classes
    'SGD',

    # Data classes
    'SyntheticRegressionDataScratch',
    'SyntheticRegressionDataTorch', 
    
    # Model classes
    'LinearRegressionScratch',
    'LinearRegressionTorch',
    'LinearRegressionTorchL2',
    'FashionMNIST',
    'SoftmaxClassifierScratch'
]