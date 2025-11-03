from .base.trainer import Trainer

from .base.evaluator import (
    Evaluator, 
    ClassificationEvaluator
)

from .base.plot import (
    plot, 
    plot_loss,
    plot_losses,
    show_images
)

from .base.utils import (
    cpu,
    gpu,
    num_gpus,
    try_gpu
)

from .regression.model import (
    LinearRegression, 
    LinearRegressionL2,
    LinearRegressionTorch
)

from .regression.dataset import (
    SyntheticRegressionDataset,
    SyntheticRegressionDatasetTorch
)

from .classification.model import (
    SoftmaxClassifier,
    SoftmaxClassifierTorch,
    MLPClassifierTorch
)

from .classification.dataset import FashionMNISTDataset