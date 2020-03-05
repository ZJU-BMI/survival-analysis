import numpy as np
import h5py
from collections import defaultdict
from sklearn.preprocessing import normalize


class DataSet(object):
    def __init__(self, dynamic_features, time,labels):
        self._dynamic_features = dynamic_features
        self._labels = labels
        self._num_examples = labels.shape[0]
        self._time = time
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
            time_res_part = self._time[start:self._num_examples]
            label_test_part = self._labels[start:self._num_examples]
            self._shuffle()
            self._index_in_epoch = 0
            return dynamic_rest_part, time_res_part,label_test_part
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._dynamic_features[start:end],self._time[start:end],self._labels[start:end]

    def predict_next_batch(self,batch_size):
        if batch_size > self._num_examples or batch_size <=0:
            batch_size = self._labels.shape[0]
        self._batch_completed += 1
        start = self._index_in_epoch
        if start + batch_size >= self._num_examples:
            self._epoch_completed += 1
            dynamic_rest_part = self._dynamic_features[start:self._num_examples]
            time_res_part = self._time[start:self._num_examples]
            label_test_part = self._labels[start:self._num_examples]
            self._index_in_epoch = 0
            return dynamic_rest_part, time_res_part,label_test_part
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._dynamic_features[start:end],self._time[start:end],self._labels[start:end]

    def _shuffle(self):
        index = np.arange(self._num_examples)
        np.random.shuffle(index)
        self._dynamic_features = self._dynamic_features[index]
        self._time = self._time[index]
        self._labels = self._labels[index]

    @property
    def dynamic_features(self):
        return self._dynamic_features

    @property
    def time(self):
        return self._time

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
        dynamic_features = np.load("embedding_features_one_year_2dims.npy")
        time = np.load("pick_4_features_one_year.npy")[:,0].reshape(-1,4,0)
        labels = np.load("pick_logistic_labels_one_year.npy")
    else:
        # dynamic_features = np.load("pick_5_features_half_year.npy")[:,:,1:]
        dynamic_features = np.load("embedding_features_half_year_2dims.npy")
        time = np.load("pick_5_features_half_year.npy")[:,:,0].reshape(-1,5,1)
        labels = np.load("pick_5_labels_half_year.npy")
    return DataSet(dynamic_features,time,labels)


# 从全部病人入院记录中平均选择5次记录（特征和标签都是加上最后一次）
def pick_5_visit():
    dynamic_fetaures = np.load("allPatientFeatures_include_id.npy")[0:2100,:,:]
    features = dynamic_fetaures[:,:,2:]
    features = features.astype(np.float64)
    labels = np.load("allPatientLabels_merge_1.npy")[0:2100, :, :]
    mask = np.sign(np.max(np.abs(features), 2))
    length = np.sum(mask, 1)
    new_features = np.zeros(shape=(0, 5, dynamic_fetaures.shape[2]))
    new_labels = np.zeros(shape=(0, 5, labels.shape[2]))
    new_stages = np.zeros(shape=(0,5,4))
    for patient in range(dynamic_fetaures.shape[0]):
        if length[patient]<6:
            one_patient_feature = dynamic_fetaures[patient,0:5,:]
            one_patient_label = labels[patient,0:5,:]

        else:
            t0 = 0
            t4 = int(length[patient] - 1)
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
    np.savetxt("new_features.csv", new_features.reshape(-1,dynamic_fetaures.shape[2]), delimiter=",",fmt='%s')
    np.savetxt("new_labels.csv", new_labels.reshape(-1,labels.shape[2]), delimiter=",",fmt='%s')
    # np.save("pick_5_visit_features_merge_1.npy", new_features)
    # np.save("pick_5_visit_labels_merge_1.npy", new_labels)


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
        dynamic_features = np.load("pick_5_visit_features_merge_1.npy")[0:2100,:,1:93].reshape(-1,92)
        # 2年
        # labels = np.load("pick_5_visit_labels_merge_1.npy")[0:2100,:,-1].reshape(-1,1)
        # 1年
        # labels = np.load("pick_5_visit_labels_merge_1.npy")[0:2100, :, -3].reshape(-1, 1)
        # 6个月
        # labels = np.load("pick_5_visit_labels_merge_1.npy")[0:2100, :, -3].reshape(-1, 1)
        # 3个月
        labels = np.load("pick_5_visit_labels_merge_1.npy")[0:2100, :, -4].reshape(-1, 1)
    else:
        # 2年
        dynamic_features = np.load("pick_5_visit_features_merge_1.npy")[0:2100,:,1:93]
        # labels = np.load("pick_5_visit_labels_merge_1.npy")[0:2100,:,-1].reshape(-1,dynamic_fetaures.shape[1],1)
        # 1年
        # labels = np.load("pick_5_visit_labels_merge_1.npy")[0:2100, :, -3].reshape(-1, dynamic_features.shape[1], 1)
        # 6个月
        # labels = np.load("pick_5_visit_labels_merge.npy")[0:2100, :, -3]
        # 3个月
        labels = np.load("pick_5_visit_labels_merge_1.npy")[0:2100, :, -3].reshape(-1,dynamic_features.shape[1],1)
        # length = np.reshape(mask,[-1,dynamic_fetaures.shape[1]])
        print(len(np.where(labels.reshape(-1,1) == 1)[0]))

    return DataSet(dynamic_features,labels)


# WHAS 数据集： trainSet 包含1310个样本 5个feature  testSet 包含328个样本 5个特征
def read_WHAS_dataset():
    filename = "metabric_IHC4_clinical_train_test.h5"
    with h5py.File(filename,'r') as file:
        datasets = defaultdict(dict)
        for key in file.keys():
            for array in file[key]:
                datasets[key][array] = file[key][array][:]

    datasets['train']['x'] = normalize(datasets['train']['x'], axis=0, norm='max').reshape(-1,1,9,1)
    datasets['test']['x'] = normalize(datasets['test']['x'], axis=0, norm='max').reshape(-1,1,9,1)

    train_e = np.array(datasets['train']['e'])
    test_e = np.array(datasets['test']['e'])
    train_time = np.array(datasets['train']['t'])
    test_time = np.array(datasets['test']['t'])

    print(np.sum(train_e))
    print(np.sum(test_e))

    censored_train_num = len(train_e)-np.sum(train_e)
    censored_test_time = len(test_e)-np.sum(test_e)

    uncensored_train_time = train_time*train_e
    uncensored_test_time = test_time*test_e

    sum_train_time = np.sum(train_time)
    sum_test_time = np.sum(test_time)

    sum_uncensored_train_time = np.sum(uncensored_train_time)
    sum_uncensored_test_time = np.sum(uncensored_test_time)

    print((sum_train_time-sum_uncensored_train_time+sum_test_time-sum_uncensored_test_time)/(censored_test_time+censored_train_num))

    return datasets


def save_time():
    data = np.load("pick_5_features_half_year.npy")[:,:,0].reshape(-1,)
    print(data.shape)
    np.savetxt("half_year_time.csv",data)

if __name__ == '__main__':
    save_time()
    read_WHAS_dataset()
    # read_data('rnn')