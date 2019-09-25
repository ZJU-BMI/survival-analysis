import numpy as np
import pandas as pd


class DataSet(object):
    def __init__(self, dynamic_features, labels):
        self._dynamic_features = dynamic_features
        self._labels = labels
        self._num_examples = labels.shape[0]
        self._epoch_completed = 0
        self._batch_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size):
        if batch_size > self._num_examples or batch_size <=0:
            batch_size = self._labels.shape[0]
        if self._batch_completed ==0:
            self._shuffle()
        self._batch_completed += 1
        start = self._index_in_epoch
        if start + batch_size >= self._num_examples:
            self._epoch_completed += 1
            dynamic_rest_part = self._dynamic_features[start:self._num_examples]
            label_test_part = self._labels[start:self._num_examples]
            self._shuffle()
            self._index_in_epoch = 0
            return dynamic_rest_part, label_test_part
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._dynamic_features[start:end],self._labels[start:end]

    def _shuffle(self):
        index = np.arange(self._num_examples)
        np.random.shuffle(index)
        self._dynamic_features = self._dynamic_features[index]
        self._labels = self._labels[index]

    @property
    def dynamic_features(self):
        return self._dynamic_features

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epoch_completed(self):
        return self._epoch_completed

    @property
    def batch_completed(self):
        return self._batch_completed

    @epoch_completed.setter
    def epoch_completed(self, value):
        self._epoch_completed = value


def read_data(name):
    if name == "LogisticRegression":
        # dynamic_features = np.load("logistic_features.npy")
        # labels = np.load("logistic_labels.npy").reshape([-1,1])
        dynamic_features = np.load("allPatientFeatures1.npy")[0:2100,0:5,:].reshape([-1,200])
        labels = np.load("allPatientLabels1.npy")[0:2100, :, -1].reshape([-1, 42, 1])[:,0:5,:].reshape([-1,1])
    else:
        dynamic_features = np.load("allPatientFeatures1.npy")
        dynamic_features = dynamic_features[0:2100,0:5,:]
        labels = np.load("allPatientLabels1.npy")[0:2100,:,-1].reshape([-1,42,1])
        labels = labels[:,0:5,:]
    return DataSet(dynamic_features,labels)
if __name__ == '__main__':
    read_data('rnn')