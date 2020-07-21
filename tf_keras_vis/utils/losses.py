import warnings

import tf_keras_vis.scores.BinaryScore as BinaryLoss  # noqa: F401
import tf_keras_vis.scores.CategoricalScore as CategoricalLoss  # noqa: F401
import tf_keras_vis.scores.InactiveScore as InactiveLoss  # noqa: F401
import tf_keras_vis.scores.Score as Loss  # noqa: F401
import tf_keras_vis.scores.SmoothedCategoricalScore as SmoothedCategoricalLoss  # noqa: F401

warnings.warn(('`tf_keras_vis.utils.losses` module is deprecated.'
               'This will be removed in future.'), DeprecationWarning)
