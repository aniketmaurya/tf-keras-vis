import warnings

from tf_keras_vis.activation_maximization.callbacks import Callback as OptimizerCallback  # noqa: F401
from tf_keras_vis.activation_maximization.callbacks import GifGenerator  # noqa: F401
from tf_keras_vis.activation_maximization.callbacks import PrintLogger as Print  # noqa: F401

warnings.warn(('`tf_keras_vis.utils.callbacks` module is deprecated.'
               'This will be removed in future.'), DeprecationWarning)
