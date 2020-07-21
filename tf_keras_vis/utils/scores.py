from abc import ABC, abstractmethod

import tensorflow as tf
import tensorflow.keras.backend as K

from tf_keras_vis.utils import listify


class Score(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def __call__(self, output):
        raise NotImplementedError()


class InactiveScore(Score):
    def __init__(self):
        super().__init__('InactiveScore')

    def __call__(self, output):
        return output * 0.0


class BinaryScore(Score):
    '''
    target_values: A bool or int value [0, 1].
    '''
    def __init__(self, target_values):
        super().__init__('BinaryScore')
        self.target_values = [bool(v) for v in listify(target_values)]

    def __call__(self, output):
        score = tf.stack([
            output[i, ...] * (1.0 if positive else -1.0)
            for i, positive in range(self.target_values)
        ])
        if len(score.shape) > 1:
            score = K.mean(score, axis=tuple(range(len(score.shape))[1:]))
        return score


class CategoricalScore(Score):
    def __init__(self, indices):
        super().__init__('CategoricalScore')
        self.indices = listify(indices)

    def __call__(self, output):
        score = tf.stack([output[i, ..., index] for i, index in enumerate(self.indices)])
        if len(score.shape) > 1:
            score = K.mean(score, axis=tuple(range(len(score.shape))[1:]))
        return score


class SmoothedCategoricalScore(Score):
    def __init__(self, indices, noise=0.05):
        super().__init__('CategoricalSmoothedScore')
        self.indices = listify(indices)
        self.noise = noise

    def __call__(self, output):
        mask = tf.fill(output.shape, (self.noise / (output.shape[-1] - 1.)))
        noise = K.mean(output * mask, axis=tuple(range(len(output.shape))[1:]))
        score = tf.stack(
            [output[i, ..., index] * (1. - self.noise) for i, index in enumerate(self.indices)])
        if len(score.shape) > 1:
            score = K.mean(score, axis=tuple(range(len(score.shape))[1:]))
        return score + noise
