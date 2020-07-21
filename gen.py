import numpy as np
import os
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16 as Model
from tf_keras_vis.activation_maximization import ActivationMaximization
from tf_keras_vis.utils.losses import CategoricalScore, SmoothedCategoricalScore
from tf_keras_vis.utils.input_modifiers import Jitter, Rotate
from tf_keras_vis.utils.regularizers import Norm, TotalVariation2D
from tf_keras_vis.utils.callbacks import Print


def gen(loss, steps, optimizer, lr, tv_weight, norm_weight, jitter, angle):
    _loss = 'CategoricalScore' if loss == CategoricalScore else 'SmoothedCategoricalScore'
    _optimizer = 'Adam' if optimizer == tf.optimizers.Adam else 'RMSprop'
    path = 'results/{}-{}-{}-{}-{}-{}-{}-{}.png'.format(_loss, steps, _optimizer, lr, tv_weight,
                                                        norm_weight, jitter, angle)
    if os.path.isfile(path):
        return
    activation = activation_maximization(
        loss(20),
        steps=steps,
        input_modifiers=[Jitter(jitter=jitter), Rotate(degree=angle)],
        regularizers=[TotalVariation2D(weight=tv_weight),
                      Norm(weight=norm_weight, p=1)],
        optimizer=optimizer(lr),
        callbacks=[Print(interval=100, prefix=path)])[0]
    image = ((activation + 1) * 127.5).astype(np.uint8)
    image = Image.fromarray(image, 'RGB')
    image.save(path)


model = Model(weights='imagenet', include_top=True)


def model_modifier(m):
    m.layers[-1].activation = tf.keras.activations.linear


activation_maximization = ActivationMaximization(model, model_modifier, clone=False)

image_titles = ['Goldfish', 'Bear', 'Assault rifle']
categories = [1, 294, 413]
seed_input = tf.random.uniform((3, 224, 224, 3), 0, 255)

for loss in [CategoricalScore, SmoothedCategoricalScore]:
    for steps in [100, 256, 512]:
        for jitter in [1, 4, 8]:
            for angle in [1, 3]:
                for tv_weight in [0, 1, 4, 8]:
                    for norm_weight in [0, 1, 4, 8]:
                        for optimizer in [tf.optimizers.RMSprop, tf.optimizers.Adam]:
                            for lr in [0.030, 0.015, 0.010, 0.008]:
                                gen(loss, steps, optimizer, lr, tv_weight, norm_weight, jitter,
                                    angle)
