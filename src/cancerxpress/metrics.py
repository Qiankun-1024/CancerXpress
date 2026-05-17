import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np


class accuracy_score(tf.keras.metrics.Metric):
    '''
        ratio of the incorrectly predicted labels over all labels
        '''

    def __init__(self, thresholds=0.5, name='cm', **kwargs):
        super(accuracy_score, self).__init__(name=name, **kwargs)
        self.thresholds = thresholds
        self.true_positives = self.add_weight(
            name='tp', shape=(1,), initializer='zeros')
        self.false_positives = self.add_weight(
            name='fp', shape=(1,), initializer='zeros')
        self.false_negatives = self.add_weight(
            name='fn', shape=(1,), initializer='zeros')
        self.true_negatives = self.add_weight(
            name='tn', shape=(1,), initializer='zeros')

    @tf.function
    def update_state(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, (-1,)), tf.float32)
        one = tf.ones_like(y_pred)
        zero = tf.zeros_like(y_pred)
        y_pred = tf.where(y_pred >= self.thresholds, x=one, y=zero)
        y_pred = tf.cast(tf.reshape(y_pred, (-1,)), tf.bool)
        for i in tf.range(tf.shape(y_true)[0]):
            if y_true[i] == 1:
                if y_pred[i]:
                    self.true_positives.assign(self.true_positives + 1.0)
                else:
                    self.false_negatives.assign(self.false_negatives + 1.0)
            if y_true[i] == 0:
                if y_pred[i]:
                    self.false_positives.assign(self.false_positives + 1.0)
                else:
                    self.true_negatives.assign(self.true_negatives + 1.0)
        return (self.true_positives, self.false_positives, self.false_negatives, self.true_negatives)

    def result(self):
        accuracy_score = tf.divide(tf.add(self.true_positives, self.true_negatives),
                                 tf.add_n([self.true_positives, self.false_positives,
                                           self.false_negatives, self.true_negatives]))
        return tf.squeeze(accuracy_score)

    def reset_states(self):
        K.batch_set_value([(v, np.zeros(1)) for v in self.variables])
        
        
        
class label_ranking_average_precision(tf.keras.metrics.Metric):
    '''
    Label ranking average precision (LRAP) is the average over each ground truth label assigned to each sample,
    evaluates the average fraction of labels ranked above ground truth label
    best value is 1
    '''
    def __init__(self, name='label_ranking_average_precision', **kwargs):
        super(label_ranking_average_precision,self).__init__(name=name, **kwargs)
        self.sum_aux = self.add_weight(
            name='sum_aux', shape=(1,), initializer='zeros')
        self.num_samples = self.add_weight(
            name='num_samples', shape=(1,), initializer='zeros')

    @tf.function
    def update_state(self, y_true, y_pred):
        n = tf.cast(tf.shape(y_true)[0], dtype=tf.float32)
        self.num_samples.assign(self.num_samples + n)

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        for i in tf.range(tf.shape(y_true)[0]):
            p_n_label = tf.where(y_true[i, :] >= 0)
            y_true_ = tf.gather_nd(y_true[i, :], p_n_label)
            y_pred_ = tf.gather_nd(y_pred[i, :], p_n_label)

            rank_index = tf.argsort(y_pred_, direction='DESCENDING')
            sort_y_true = tf.gather(y_true_, rank_index)
            sort_y_pred = tf.gather(y_pred_, rank_index)
            truth_index = tf.where(sort_y_true==1)
            pred_rank = tf.argsort(sort_y_pred, direction='DESCENDING') + 1
            truth_rank = tf.gather(pred_rank, truth_index)
            L = tf.argsort(truth_rank, axis=0) + 1
            aux = tf.cond(tf.shape(L)[0] != 0, lambda :tf.cast(tf.reduce_mean(L / truth_rank), tf.float32)
                          , lambda :tf.constant(1.))    #true label rank in all true labels/ true label rank in all labels
            self.sum_aux.assign(self.sum_aux + aux)

        return self.num_samples, self.sum_aux

    def result(self):
        out = tf.divide(self.sum_aux, self.num_samples)
        return tf.squeeze(out)

    def reset_states(self):
        K.batch_set_value([(v, np.zeros(1)) for v in self.variables])



class micro_average(tf.keras.metrics.Metric):

    def __init__(self, thresholds=0.5, name='micro_average', **kwargs):
        super(micro_average, self).__init__(name=name, **kwargs)
        self.thresholds = thresholds
        self.true_positives = self.add_weight(
            name='tp', shape=(1,), initializer='zeros')
        self.false_positives = self.add_weight(
            name='fp', shape=(1,), initializer='zeros')
        self.false_negatives = self.add_weight(
            name='fn', shape=(1,), initializer='zeros')
        self.true_negatives = self.add_weight(
            name='tn', shape=(1,), initializer='zeros')

    # @tf.function
    @tf.function(experimental_relax_shapes=True)
    def update_state(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, (-1,)), tf.float32)
        one = tf.ones_like(y_pred)
        zero = tf.zeros_like(y_pred)
        y_pred = tf.where(y_pred >= self.thresholds, x=one, y=zero)
        y_pred = tf.cast(tf.reshape(y_pred, (-1,)), tf.bool)
        for i in tf.range(tf.shape(y_true)[0]):
            if y_true[i] == 1:
                if y_pred[i]:
                    self.true_positives.assign(self.true_positives + 1.0)
                else:
                    self.false_negatives.assign(self.false_negatives + 1.0)
            if y_true[i] == 0:
                if y_pred[i]:
                    self.false_positives.assign(self.false_positives + 1.0)
                else:
                    self.true_negatives.assign(self.true_negatives + 1.0)
        return (self.true_positives, self.false_positives, self.false_negatives, self.true_negatives)

    def result(self):
        micro_precision = tf.divide(self.true_positives,
                                    tf.add_n([self.true_positives, self.false_positives, tf.constant([0.000001])]))
        micro_racall = tf.divide(self.true_positives,
                                    tf.add_n([self.true_positives, self.false_negatives, tf.constant([0.000001])]))
        micro_f1 = tf.divide((tf.constant(2.0) * micro_precision * micro_racall),
                             (micro_precision+ micro_racall+ tf.constant([0.000001])))
        micro_score = (micro_precision,micro_racall,micro_f1)
        return tf.squeeze(micro_score)

    def reset_states(self):
        K.batch_set_value([(v, np.zeros(1)) for v in self.variables])


