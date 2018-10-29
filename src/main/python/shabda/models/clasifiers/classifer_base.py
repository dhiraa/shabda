import tensorflow as tf
from overrides import overrides
from shabda.models.internal.model_base import ModelBase

class ClassifierBase(ModelBase):
    def __init__(self, hparams):
        ModelBase.__init__(self, hparams=hparams)

        self._out_dim = self._hparams.out_dim
        self._learning_rate = self.hparams.learning_rate
        # self._layers = {}

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.
        """
        hparams = {
            "name": "classifier_base",
            "out_dim" : -1,
            "learning_rate" : 0.001
        }
        return hparams

    def _get_loss(self, labels, logits):
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=labels,
            logits=logits,
            scope='softmax_cross_entropy_loss')

        loss = tf.reduce_mean(loss, name="softmax_cross_entropy_mean_loss")
        return loss

    def _build_layers(self, features):
        layer1 = tf.layers.dense(inputs=features,
                                 units=self._out_dim,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(seed=42))
        layer2 = tf.layers.dense(inputs=layer1,
                                 units=self._out_dim,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(seed=42))
        logits = tf.layers.dense(inputs=layer2,
                                 units=self._out_dim,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(seed=42))
        return logits

    def _get_predicted_classes(self, logits):
        predicted_class = tf.argmax(logits, axis=1, name="class_output")
        tf.logging.info('predicted_class: -----> {}'.format(predicted_class))
        return predicted_class
    
    def _get_class_probabilities(self, logits):
        predicted_probabilities = tf.nn.softmax(logits, name="softmax_output")
        tf.logging.info('predicted_probabilities: -----> {}'.format(predicted_probabilities))
        return predicted_probabilities

    def _get_optimizer(self, loss):
        optimizer = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            optimizer=tf.train.AdamOptimizer,
            learning_rate=self._learning_rate)
        return optimizer

    def _get_eval_metrics(self, labels, logits):
        label_argmax = tf.argmax(labels, 1, name='label_argmax')
        predicted_class = self._get_predicted_classes(logits=logits)
        
        eval_metric_ops = {
            'Accuracy': tf.metrics.accuracy(
                labels=label_argmax,
                predictions=predicted_class,
                name='accuracy'),
            'Precision': tf.metrics.precision(
                labels=label_argmax,
                predictions=predicted_class,
                name='Precision'),
            'Recall': tf.metrics.recall(
                labels=label_argmax,
                predictions=predicted_class,
                name='Recall')
        }
        return eval_metric_ops

    @overrides
    def _build(self, features, labels, params, mode, config=None):

        # Loss, training and eval operations are not needed during inference.
        loss = None
        optimizer = None
        eval_metric_ops = {}

        logits = self._build_layers(features=features)
        predicted_class = self._get_predicted_classes(logits=logits)
        predicted_probabilities = self._get_class_probabilities(logits=logits)
        predictions = {
            "classes": predicted_class,
            "probabilities": predicted_probabilities
        }

        if mode != tf.estimator.ModeKeys.PREDICT:
            loss = self._get_loss(labels=labels, logits=logits)
            optimizer = self._get_optimizer(loss)
            eval_metric_ops = self._get_eval_metrics(logits=logits, labels=labels)


        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=optimizer,
            eval_metric_ops=eval_metric_ops)