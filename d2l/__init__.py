from .base.trainer import (
    TrainerType,
    Trainer
)

from .base.evaluator import (
    EvaluatorType,
    Evaluator, 
    ClassificationEvaluator
)

from .base.plot import (
    plot, 
    plot_loss,
    plot_losses,
    show_images
)

from .base.function import (
    softmax,
    log_sum_exp 
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
    MLPClassifier,
    MLPClassifierTorch
)

from .classification.dataset import FashionMNISTDataset