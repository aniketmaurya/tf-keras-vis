import warnings

import tf_keras_vis.activation_maximization.callbacks.Callback as OptimizerCallback  # noqa: F401
import tf_keras_vis.activation_maximization.callbacks.GifGenerator  # noqa: F401
import tf_keras_vis.activation_maximization.callbacks.PrintLogger as Print  # noqa: F401

warnings.warn(('`tf_keras_vis.utils.callbacks` module is deprecated.'
               'This will be removed in future.'), DeprecationWarning)
