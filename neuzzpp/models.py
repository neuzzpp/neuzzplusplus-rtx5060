# Copyright (c) 2023 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Module containing neural network architectures."""
from typing import List, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class MLP:
    """
    Basic MLP with one hidden layer as used in the original NEUZZ implementation.

    Output is sigmoid (multi-label bitmap).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        lr: float,
        ff_dim: int = 4096,
        output_bias: Optional[float] = None,
        fast: bool = False,
    ) -> None:
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

        bias_init = None
        if output_bias is not None:
            bias_init = tf.keras.initializers.Constant(float(output_bias))

        # Keras-3-safe model construction: define Input explicitly
        inputs = tf.keras.Input(shape=(self.input_dim,), dtype=tf.float32, name="seed_bytes")
        x = layers.Dense(ff_dim, activation="relu")(inputs)
        logits = layers.Dense(self.output_dim, bias_initializer=bias_init, name="logits")(x)
        outputs = layers.Activation("sigmoid")(logits)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="neuzzpp_mlp")

        # Compile
        lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(lr, first_decay_steps=1000)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sched)

        metrics = [
            tf.keras.metrics.AUC(
                name="prc", curve="PR", multi_label=True, num_labels=self.output_dim
            )
        ]
        if not fast:
            metrics.extend(
                [
                    tf.keras.metrics.BinaryAccuracy(name="acc"),
                    tf.keras.metrics.AUC(name="auc", multi_label=True, num_labels=self.output_dim),
                ]
            )

        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=metrics,
        )
        self.model = model


def create_logits_model(model: tf.keras.Model) -> tf.keras.Model:
    """
    Create a model that outputs logits (pre-sigmoid) for gradient computation.
    Works reliably with Keras 2.x and 3.x.
    """
    try:
        logits_layer = model.get_layer("logits")
    except Exception as e:
        raise ValueError("Expected a layer named 'logits' in the model, but it was not found.") from e

    # Use model.inputs (list) - stable in Keras 3
    return tf.keras.Model(inputs=model.inputs, outputs=logits_layer.output, name="neuzzpp_logits")


def predict_coverage(model: tf.keras.Model, inputs: List[np.ndarray]) -> np.ndarray:
    """
    Get binary labels from model for non-normalized input data.
    """
    input_shape = int(model.inputs[0].shape[-1])
    inputs_preproc = tf.keras.preprocessing.sequence.pad_sequences(
        inputs, padding="post", dtype="float32", maxlen=input_shape
    )
    inputs_preproc = inputs_preproc.astype("float32") / 255.0

    preds = model(inputs_preproc).numpy()
    return preds > 0.5
