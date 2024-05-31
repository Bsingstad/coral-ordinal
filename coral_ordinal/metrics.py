import tensorflow as tf
from tensorflow.python.keras.metrics import metrics_utils
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras import initializers
import keras
from keras import ops

from . import activations

def _label_to_levels(labels: tf.Tensor, num_classes: int) -> tf.Tensor:
    # Original code that we are trying to replicate:
    # levels = [1] * label + [0] * (self.num_classes - 1 - label)
    # This function uses tf.sequence_mask(), which is vectorized. Avoids map_fn()
    # call.
    return tf.sequence_mask(labels, maxlen=num_classes - 1, dtype=tf.float32)



@tf.keras.utils.register_keras_serializable(package="coral_ordinal")
class MeanAbsoluteErrorLabels(tf.keras.metrics.Metric):
    """Computes mean absolute error for ordinal labels."""

    def __init__(
        self,
        corn_logits: bool = False,
        threshold: float = 0.5,
        name="mean_absolute_error_labels",
        **kwargs
    ):
        """Creates a `MeanAbsoluteErrorLabels` instance.

        Args:
          corn_logits: if True, inteprets y_pred as CORN logits; otherwise (default)
            as CORAL logits.
          threshold: which threshold should be used to determine the label from
            the cumulative probabilities. Defaults to 0.5.
          name: name of metric.
          **kwargs: keyword arguments passed to parent Metric().
        """
        super().__init__(name=name, **kwargs)
        self._corn_logits = corn_logits
        self._threshold = threshold
        self.maes = self.add_weight(name="maes", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Computes mean absolute error for ordinal labels.

        Args:
          y_true: Labels (int).
          y_pred: Cumulative logits from CoralOrdinal layer.
          sample_weight (optional): sample weights to weight absolute error.
        """

        # Predict the label as in Cao et al. - using cumulative probabilities.
        if self._corn_logits:
            cumprobs = activations.corn_cumprobs(y_pred)
        else:
            cumprobs = activations.coral_cumprobs(y_pred)

        # Threshold cumulative probabilities at predefined cutoff (user set).
        label_pred = tf.cast(
            activations.cumprobs_to_label(cumprobs, threshold=self._threshold),
            dtype=tf.float32,
        )
        y_true = tf.cast(y_true, label_pred.dtype)

        # remove all dimensions of size 1 (e.g., from [[1], [2]], to [1, 2])
        y_true = tf.squeeze(y_true)
        label_pred = tf.squeeze(label_pred)
        label_abs_err = tf.abs(y_true - label_pred)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, y_true.dtype)
            sample_weight = tf.broadcast_to(sample_weight, label_abs_err.shape)
            label_abs_err = tf.multiply(label_abs_err, sample_weight)

        self.maes.assign_add(tf.reduce_mean(label_abs_err))
        self.count.assign_add(tf.constant(1.0))

    def result(self):
        return tf.math.divide_no_nan(self.maes, self.count)

    def reset_state(self):
        """Resets all of the metric state variables at the start of each epoch."""
        K.batch_set_value([(v, 0) for v in self.variables])

    def get_config(self):
        """Returns the serializable config of the metric."""
        config = {"threshold": self._threshold, "corn_logits": self._corn_logits}
        base_config = super().get_config()
        return {**base_config, **config}


@tf.keras.utils.register_keras_serializable(package="coral_ordinal")
class AUROC(tf.keras.metrics.Metric):
    """Computes mean absolute error for ordinal labels."""

    def __init__(
        self,
        num_thresholds=200,
        curve="ROC",
        summation_method="interpolation",
        name=None,
        dtype=None,
        thresholds=None,
        multi_label=False,
        num_labels=None,
        label_weights=None,
        from_logits=False,
    ):
        """
        AUC
        """
                # Metric should be maximized during optimization.
        self._direction = "up"

        # Validate configurations.
        if isinstance(curve, metrics_utils.AUCCurve) and curve not in list(
            metrics_utils.AUCCurve
        ):
            raise ValueError(
                f'Invalid `curve` argument value "{curve}". '
                f"Expected one of: {list(metrics_utils.AUCCurve)}"
            )
        if isinstance(
            summation_method, metrics_utils.AUCSummationMethod
        ) and summation_method not in list(metrics_utils.AUCSummationMethod):
            raise ValueError(
                "Invalid `summation_method` "
                f'argument value "{summation_method}". '
                f"Expected one of: {list(metrics_utils.AUCSummationMethod)}"
            )

        # Update properties.
        self._init_from_thresholds = thresholds is not None
        if thresholds is not None:
            # If specified, use the supplied thresholds.
            self.num_thresholds = len(thresholds) + 2
            thresholds = sorted(thresholds)
            self._thresholds_distributed_evenly = (
                metrics_utils.is_evenly_distributed_thresholds(
                    np.array([0.0] + thresholds + [1.0])
                )
            )
        else:
            if num_thresholds <= 1:
                raise ValueError(
                    "Argument `num_thresholds` must be an integer > 1. "
                    f"Received: num_thresholds={num_thresholds}"
                )

            # Otherwise, linearly interpolate (num_thresholds - 2) thresholds in
            # (0, 1).
            self.num_thresholds = num_thresholds
            thresholds = [
                (i + 1) * 1.0 / (num_thresholds - 1)
                for i in range(num_thresholds - 2)
            ]
            self._thresholds_distributed_evenly = True

        # Add an endpoint "threshold" below zero and above one for either
        # threshold method to account for floating point imprecisions.
        self._thresholds = np.array(
            [0.0 - K.epsilon()] + thresholds + [1.0 + K.epsilon()]
        )

        if isinstance(curve, metrics_utils.AUCCurve):
            self.curve = curve
        else:
            self.curve = metrics_utils.AUCCurve.from_str(curve)
        if isinstance(summation_method, metrics_utils.AUCSummationMethod):
            self.summation_method = summation_method
        else:
            self.summation_method = metrics_utils.AUCSummationMethod.from_str(
                summation_method
            )
        super().__init__(name=name, dtype=dtype)

                # Handle multilabel arguments.
        self.multi_label = multi_label
        self.num_labels = num_labels
        if label_weights is not None:
            label_weights = ops.array(label_weights, dtype=self.dtype)
            self.label_weights = label_weights

        else:
            self.label_weights = None

        self._from_logits = from_logits

        self._built = False
        if self.multi_label:
            if num_labels:
                shape = [None, num_labels]
                self._build(shape)
        else:
            if num_labels:
                raise ValueError(
                    "`num_labels` is needed only when `multi_label` is True."
                )
            self._build(None)

    @property
    def thresholds(self):
        """The thresholds used for evaluating AUC."""
        return list(self._thresholds)

    def _build(self, shape):
        """Initialize TP, FP, TN, and FN tensors, given the shape of the
        data."""
        if self.multi_label:
            if len(shape) != 2:
                raise ValueError(
                    "`y_pred` must have rank 2 when `multi_label=True`. "
                    f"Found rank {len(shape)}. "
                    f"Full shape received for `y_pred`: {shape}"
                )
            self._num_labels = shape[1]
            variable_shape = [self.num_thresholds, self._num_labels]
        else:
            variable_shape = [self.num_thresholds]

        self._build_input_shape = shape
        # Create metric variables
        self.true_positives = self.add_variable(
            shape=variable_shape,
            initializer=initializers.Zeros(),
            name="true_positives",
        )
        self.false_positives = self.add_variable(
            shape=variable_shape,
            initializer=initializers.Zeros(),
            name="false_positives",
        )
        self.true_negatives = self.add_variable(
            shape=variable_shape,
            initializer=initializers.Zeros(),
            name="true_negatives",
        )
        self.false_negatives = self.add_variable(
            shape=variable_shape,
            initializer=initializers.Zeros(),
            name="false_negatives",
        )

        self._built = True


    def update_state(self, y_true, y_pred, sample_weight=None):
        """Computes mean absolute error for ordinal labels.

        Args:
          y_true: Labels (int).
          y_pred: Cumulative logits from CoralOrdinal layer.
          sample_weight (optional): sample weights to weight absolute error.
        """

        # Predict the label as in Cao et al. - using cumulative probabilities.
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        
        y_true = _label_to_levels(tf.squeeze(y_true), self.num_classes)
        if not self._built:
            self._build(y_pred.shape)

        if self.multi_label or (self.label_weights is not None):
            # y_true should have shape (number of examples, number of labels).
            shapes = [(y_true, ("N", "L"))]
            if self.multi_label:
                # TP, TN, FP, and FN should all have shape
                # (number of thresholds, number of labels).
                shapes.extend(
                    [
                        (self.true_positives, ("T", "L")),
                        (self.true_negatives, ("T", "L")),
                        (self.false_positives, ("T", "L")),
                        (self.false_negatives, ("T", "L")),
                    ]
                )
            if self.label_weights is not None:
                # label_weights should be of length equal to the number of
                # labels.
                shapes.append((self.label_weights, ("L",)))

        # Only forward label_weights to update_confusion_matrix_variables when
        # multi_label is False. Otherwise the averaging of individual label AUCs
        # is handled in AUC.result
        label_weights = None if self.multi_label else self.label_weights

        if self._from_logits:
            y_pred = activations.sigmoid(y_pred)

        metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,  # noqa: E501
                metrics_utils.ConfusionMatrix.TRUE_NEGATIVES: self.true_negatives,  # noqa: E501
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,  # noqa: E501
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives,  # noqa: E501
            },
            y_true,
            y_pred,
            self._thresholds,
            thresholds_distributed_evenly=self._thresholds_distributed_evenly,
            sample_weight=sample_weight,
            multi_label=self.multi_label,
            label_weights=label_weights,
        )




    def result(self):
        return tf.math.divide_no_nan(self.maes, self.count)

    def reset_state(self):
        """Resets all of the metric state variables at the start of each epoch."""
        K.batch_set_value([(v, 0) for v in self.variables])

    def get_config(self):
        """Returns the serializable config of the metric."""
        config = {"threshold": self._threshold, "corn_logits": self._corn_logits}
        base_config = super().get_config()
        return {**base_config, **config}


    def result(self):
        # Set `x` and `y` values for the curves based on `curve` config.
        recall = ops.divide_no_nan(
            self.true_positives,
            ops.add(self.true_positives, self.false_negatives),
        )
        if self.curve == metrics_utils.AUCCurve.ROC:
            fp_rate = ops.divide_no_nan(
                self.false_positives,
                ops.add(self.false_positives, self.true_negatives),
            )
            x = fp_rate
            y = recall
        else:  # curve == 'PR'.
            precision = ops.divide_no_nan(
                self.true_positives,
                ops.add(self.true_positives, self.false_positives),
            )
            x = recall
            y = precision

        # Find the rectangle heights based on `summation_method`.
        if (
            self.summation_method
            == metrics_utils.AUCSummationMethod.INTERPOLATION
        ):
            # Note: the case ('PR', 'interpolation') has been handled above.
            heights = ops.divide(
                ops.add(y[: self.num_thresholds - 1], y[1:]), 2.0
            )
        elif self.summation_method == metrics_utils.AUCSummationMethod.MINORING:
            heights = ops.minimum(y[: self.num_thresholds - 1], y[1:])
        # self.summation_method = metrics_utils.AUCSummationMethod.MAJORING:
        else:
            heights = ops.maximum(y[: self.num_thresholds - 1], y[1:])

        # Sum up the areas of all the rectangles.
        riemann_terms = ops.multiply(
            ops.subtract(x[: self.num_thresholds - 1], x[1:]), heights
        )
        if self.multi_label:
            by_label_auc = ops.sum(riemann_terms, axis=0)

            if self.label_weights is None:
                # Unweighted average of the label AUCs.
                return ops.mean(by_label_auc)
            else:
                # Weighted average of the label AUCs.
                return ops.divide_no_nan(
                    ops.sum(ops.multiply(by_label_auc, self.label_weights)),
                    ops.sum(self.label_weights),
                )
        else:
            return ops.sum(riemann_terms)

    def reset_state(self):
        if self._built:
            if self.multi_label:
                variable_shape = (self.num_thresholds, self._num_labels)
            else:
                variable_shape = (self.num_thresholds,)

            self.true_positives.assign(ops.zeros(variable_shape))
            self.false_positives.assign(ops.zeros(variable_shape))
            self.true_negatives.assign(ops.zeros(variable_shape))
            self.false_negatives.assign(ops.zeros(variable_shape))

    def get_config(self):
        label_weights = self.label_weights
        config = {
            "num_thresholds": self.num_thresholds,
            "curve": self.curve.value,
            "summation_method": self.summation_method.value,
            "multi_label": self.multi_label,
            "num_labels": self.num_labels,
            "label_weights": label_weights,
            "from_logits": self._from_logits,
        }
        # optimization to avoid serializing a large number of generated
        # thresholds
        if self._init_from_thresholds:
            # We remove the endpoint thresholds as an inverse of how the
            # thresholds were initialized. This ensures that a metric
            # initialized from this config has the same thresholds.
            config["thresholds"] = self.thresholds[1:-1]
        base_config = super().get_config()
        return {**base_config, **config}
