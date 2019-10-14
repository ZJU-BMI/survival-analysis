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


# 从全部病人入院记录中平均选择5次记录（特征和标签都是加上最后一次）
def pick_5_visit():
    dynamic_fetaures = np.load("allPatientFeatures_merge.npy")[0:2100,:,0:]
    labels = np.load("allPatientLabels_merge.npy")[0:2100, :, -1].reshape(-1,dynamic_fetaures.shape[1],1)
    mask = np.sign(np.max(np.abs(dynamic_fetaures), 2))
    length = np.sum(mask, 1)
    new_features = np.zeros(shape=(0, 5, dynamic_fetaures.shape[2]))
    new_labels = np.zeros(shape=(0, 5, labels.shape[2]))
    for patient in range(dynamic_fetaures.shape[0]):
        if length[patient]<6:
            one_patient_feature = dynamic_fetaures[patient,0:5,:]
            one_patient_label = labels[patient,0:5,:]
        else:
            t0 = 0
            t4 = length[patient] - 1
            t2 = int(length[patient] / 2)
            t1 = int((t0 + t2) / 2)
            t3 = int((t2 + t4) / 2)
            t = [t0, t1, t2, t3, t4]
            one_patient_feature = np.zeros(shape=(0, dynamic_fetaures.shape[2]))
            one_patient_label = np.zeros(shape=(0, labels.shape[2]))
            for T in range(5):
                one_visit_feature = dynamic_fetaures[patient, t[T], :].reshape(-1,dynamic_fetaures.shape[2])
                one_patient_feature = np.concatenate((one_patient_feature, one_visit_feature))
                one_visit_label = labels[patient, t[T], :].reshape(-1,labels.shape[2])
                one_patient_label = np.concatenate((one_patient_label, one_visit_label))
            print(one_patient_label)
            print(one_patient_feature)
        one_patient_fetaure = one_patient_feature.reshape(-1, 5, dynamic_fetaures.shape[2])
        new_features = np.concatenate((new_features, one_patient_fetaure))
        one_patient_label = one_patient_label.reshape([-1, 5, labels.shape[2]])
        new_labels = np.concatenate((new_labels, one_patient_label))
    np.save("pick_5_visit_features_merge.npy", new_features)
    np.save("pick_5_visit_labels_merge.npy", new_labels)


# 将特征去除心功能 和 时间差
def get_195_features():
    features = np.load("pick_5_visit_features.npy")
    time = features[:, :, 10]
    features1 = features[:, :, 0:3]
    features2 = features[:, :, 7:10]
    features3 = features[:, :, 11:]
    features_concentrate = np.concatenate((features1, np.concatenate((features2, features3), axis=2)), axis=2)
    print(features_concentrate.shape)
    np.save("pick_5_visit_features_195.npy", features_concentrate)


def get_pick_data(name):
    if name == 'LogisticRegression':
        dynamic_fetaures = np.load("pick_5_visit_features_merge.npy")[0:2100,:,1:].reshape(-1,94)
        labels = np.load("pick_5_visit_labels_merge.npy")[0:2100,:,-1].reshape(-1,1)
    else:
        dynamic_fetaures = np.load("pick_5_visit_features_merge.npy")[0:2100,:,1:]
        labels = np.load("pick_5_visit_labels_merge.npy")[0:2100,:,:]
        # length = np.reshape(mask,[-1,dynamic_fetaures.shape[1]])
    return DataSet(dynamic_fetaures,labels)

if __name__ == '__main__':
    pick_5_visit()
    # get_195_features()