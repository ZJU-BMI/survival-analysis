import os
import jenkspy
import matplotlib.pyplot as plt
import sklearn
import time
import xlwt
from lifelines import CoxPHFitter
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from lifelines.utils import k_fold_cross_validation, concordance_index
from sklearn.cluster import KMeans
from random_survival_forest import RandomSurvivalForest
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, roc_curve
from data import get_pick_data, DataSet,read_WHAS_dataset,read_data
from models import BidirectionalLSTMModel, AttentionLSTMModel, LogisticRegression, SelfAttentionLSTMModel
from GNN_LSTM import GnnLSTMSurV
import random
from tensorflow.python.tools import inspect_checkpoint as ickpt

class ExperimentSetup(object):
    kfold = 5  # 5折交叉验证
    output_n_epochs = 1

    def __init__(self, learning_rate, max_loss=2.0, max_pace=0.01, ridge=0.0, batch_size=64, hidden_size=2, epoch=100,
                 dropout=1.0):
        self._learning_rate = learning_rate
        self._max_loss = max_loss
        self._max_pace = max_pace
        self._ridge = ridge
        self._batch_size = batch_size
        self._hidden_size = hidden_size
        self._epoch = epoch
        self._dropout = dropout

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def max_loss(self):
        return self._max_loss

    @property
    def max_pace(self):
        return self._max_pace

    @property
    def ridge(self):
        return self._ridge

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def epoch(self):
        return self._epoch

    @property
    def dropout(self):
        return self._dropout

    @property
    def all(self):
        return self._learning_rate, self._max_loss, self._max_pace, self._ridge, self._batch_size, \
               self._hidden_size, self._epoch, self._dropout


# set the parameters
lr_setup = ExperimentSetup(0.01, 2, 0.0001, 0.0001)
bi_lstm_setup = ExperimentSetup(0.001, 0.08, 0.01, 0.05)
global_rnn_setup = ExperimentSetup(0.00005, 0.08, 0.001, 0.0001)
self_rnn_setup = ExperimentSetup(0.0001, 0.08, 0.001, 0.001)
# mini_batch_list = [32,64,128,256]
# lr = 10 ** random.randint(-5,1)
# ridge_l2 = float(10 ** random.randint(-3,1))
# mini_batch = mini_batch_list[np.random.randint(0,4)]
# epochs = random.randint(10,300)
gnn_lstm_setup = ExperimentSetup(0.01, 0.08, 0.01, 0.01,64,2,100,1.0)


def split_data_set(dynamic_features, time, labels):
    time_steps = dynamic_features.shape[1]
    num_features = dynamic_features.shape[2]
    # feature_dims = dynamic_features.shape[3]
    train_dynamic_features = {}
    train_labels = {}
    train_time = {}
    test_dynamic_features = {}
    test_time = {}
    test_labels = {}
    num = int(dynamic_features.shape[0] / 5)
    for i in range(4):
        test_dynamic_features[i] = dynamic_features[i * num:(i + 1) * num, :, :].reshape(-1, time_steps, num_features)
        test_time[i] = time[i * num:(i + 1) * num, :, :].reshape(-1,time_steps,1)
        test_labels[i] = labels[i * num:(i + 1) * num, :, :].reshape(-1, time_steps, 1)

    test_dynamic_features[4] = dynamic_features[4*num:,:,:].reshape(-1,time_steps,num_features)
    test_time[4] = time[4*num:,:,:].reshape(-1,time_steps,1)
    test_labels[4] = labels[4*num:,:,:].reshape(-1, time_steps,1)

    train_dynamic_features[0] = dynamic_features[num:, :, :]
    train_time[0] = time[num:,:,:].reshape(-1,time_steps,1)
    train_labels[0] = labels[num:, :, :]

    train_dynamic_features[1] = np.vstack((dynamic_features[0:num, :, :], dynamic_features[2 * num:, :, :]))
    train_time[1] = np.vstack((time[0:num, :, :], time[2 * num:, :, :])).reshape(-1,time_steps,1)
    train_labels[1] = np.vstack((labels[0:num, :, :], labels[2 * num:, :, :]))

    train_dynamic_features[2] = np.vstack((dynamic_features[0:2 * num, :, :], dynamic_features[3 * num:, :, :]))
    train_time[2] = np.vstack((time[0:2 * num, :, :], time[3 * num:, :, :])).reshape(-1,time_steps,1)
    train_labels[2] = np.vstack((labels[0:2 * num, :, :], labels[3 * num:, :, :]))

    train_dynamic_features[3] = np.vstack((dynamic_features[0:3 * num, :, :], dynamic_features[4 * num:, :, :]))
    train_time[3] = np.vstack((time[0:3 * num, :, :], time[4 * num:, :, :])).reshape(-1,time_steps,1)
    train_labels[3] = np.vstack((labels[0:3 * num, :, :], labels[4 * num:, :, :]))

    train_dynamic_features[4] = dynamic_features[0:4 * num, :, :]
    train_time[4] = time[0:4 * num, :, :].reshape(-1,time_steps,1)
    train_labels[4] = labels[0:4 * num, :, :]

    return train_dynamic_features, test_dynamic_features, train_time, test_time,train_labels, test_labels


def split_data_set_gnn(dynamic_features, time, labels):
    time_steps = dynamic_features.shape[1]
    num_features = dynamic_features.shape[2]
    feature_dims = dynamic_features.shape[3]
    train_dynamic_features = {}
    train_labels = {}
    train_time = {}
    test_dynamic_features = {}
    test_time = {}
    test_labels = {}
    num = int(dynamic_features.shape[0] / 5)
    for i in range(4):
        test_dynamic_features[i] = dynamic_features[i * num:(i + 1) * num, :, :,:].reshape(-1, time_steps, num_features,feature_dims)
        test_time[i] = time[i * num:(i + 1) * num, :, :].reshape(-1,time_steps,1)
        test_labels[i] = labels[i * num:(i + 1) * num, :, :].reshape(-1, time_steps, 1)

    test_dynamic_features[4] = dynamic_features[4*num:,:,:,:].reshape(-1,time_steps,num_features,feature_dims)
    test_time[4] = time[4*num:,:,:].reshape(-1,time_steps,1)
    test_labels[4] = labels[4*num:,:,:].reshape(-1, time_steps,1)

    train_dynamic_features[0] = dynamic_features[num:, :, :,:]
    train_time[0] = time[num:,:,:].reshape(-1,time_steps,1)
    train_labels[0] = labels[num:, :, :]

    train_dynamic_features[1] = np.vstack((dynamic_features[0:num, :, :,:], dynamic_features[2 * num:, :, :,:]))
    train_time[1] = np.vstack((time[0:num, :, :], time[2 * num:, :, :])).reshape(-1,time_steps,1)
    train_labels[1] = np.vstack((labels[0:num, :, :], labels[2 * num:, :, :]))

    train_dynamic_features[2] = np.vstack((dynamic_features[0:2 * num, :, :,:], dynamic_features[3 * num:, :, :,:]))
    train_time[2] = np.vstack((time[0:2 * num, :, :], time[3 * num:, :, :])).reshape(-1,time_steps,1)
    train_labels[2] = np.vstack((labels[0:2 * num, :, :], labels[3 * num:, :, :]))

    train_dynamic_features[3] = np.vstack((dynamic_features[0:3 * num, :, :,:], dynamic_features[4 * num:, :, :,:]))
    train_time[3] = np.vstack((time[0:3 * num, :, :], time[4 * num:, :, :])).reshape(-1,time_steps,1)
    train_labels[3] = np.vstack((labels[0:3 * num, :, :], labels[4 * num:, :, :]))

    train_dynamic_features[4] = dynamic_features[0:4 * num, :, :,:]
    train_time[4] = time[0:4 * num, :, :].reshape(-1,time_steps,1)
    train_labels[4] = labels[0:4 * num, :, :]

    return train_dynamic_features, test_dynamic_features, train_time, test_time,train_labels, test_labels

def split_logistic_data(dynamic_features, labels):
    train_dynamic_features = {}
    train_labels = {}
    test_dynamic_features = {}
    test_labels = {}
    num = int(dynamic_features.shape[0] / 5)

    for i in range(4):
        test_dynamic_features[i] = dynamic_features[num * i:num * (i + 1), :]
        test_labels[i] = labels[num * i:num * (i + 1), :]

    test_dynamic_features[4] = dynamic_features[num*4:,:]
    test_labels[4] = labels[num*4:,:]

    train_dynamic_features[0] = dynamic_features[num:, :]
    train_labels[0] = labels[num:, :]

    train_dynamic_features[1] = np.vstack((dynamic_features[2 * num:, :], dynamic_features[0:num, :]))
    train_labels[1] = np.vstack((labels[2 * num:, :], labels[0:num, :]))

    train_dynamic_features[2] = np.vstack((dynamic_features[3 * num: :], dynamic_features[0:2 * num, :]))
    train_labels[2] = np.vstack((labels[3 * num:, :], labels[0:2 * num, :]))

    train_dynamic_features[3] = np.vstack((dynamic_features[4 * num:, :], dynamic_features[0:3 * num, :]))
    train_labels[3] = np.vstack((labels[4 * num:, :], labels[0:3 * num, :]))

    train_dynamic_features[4] = dynamic_features[0:4 * num, :]
    train_labels[4] = labels[0:4 * num]

    return train_dynamic_features, test_dynamic_features, train_labels, test_labels


def evaluate(test_index, y_label, y_score, file_name):
    """

    :param test_index: sample index of the test_set
    :param y_label:  the label of test_set
    :param y_score: the prediction of test_set
    :param file_name: path of the output
    """
    wb = xlwt.Workbook(file_name + ".xls")
    table = wb.add_sheet('Sheet1')
    table_title = ["test_index", "label", "prob", "pre", " ", "fpr", "tpr", "thresholds", " ", "fp", "tp", "fn", "tn",
                   "fp_words", "fp_freq", "tp_words", "tp_freq", "fn_words", "fn_freq", "tn_words", "tn_freq", " ",
                   "acc", "auc", "recall", "precision", "f1-score", "threshold"]
    for i in range(len(table_title)):
        table.write(0, i, table_title[i])
    # y_label = y_label.reshape([-1, 1])
    # y_score = y_score.reshape([-1, 1])
    auc = roc_auc_score(y_label, y_score)
    threshold = plot_roc(y_label, y_score, table, table_title, file_name)
    y_pred_label = (y_score >= threshold) * 1
    acc = accuracy_score(y_label, y_pred_label)
    recall = recall_score(y_label, y_pred_label)
    precision = precision_score(y_label, y_pred_label)
    f1 = f1_score(y_label, y_pred_label)

    # write metrics
    table.write(1, table_title.index("auc"), float(auc))
    table.write(1, table_title.index("acc"), float(acc))
    table.write(1, table_title.index("recall"), float(recall))
    table.write(1, table_title.index("precision"), float(precision))
    table.write(1, table_title.index("f1-score"), float(f1))

    # collect samples of FP,TP,FN,TP and write the result
    fp_samples = []
    fn_samples = []
    tp_samples = []
    tn_samples = []
    fp_count = 1
    tp_count = 1
    tn_count = 1
    fn_count = 1
    all_samples = read_data("GNN").dynamic_features
    # all_samples = all_samples.reshape(-1,all_samples.shape[3]*all_samples.shape[2])
    all_samples = all_samples.reshape(-1,all_samples.shape[2])
    for j in range(len(y_label)):
        if y_label[j] == 0 and y_pred_label[j] == 1:  # FP
            write_result(j, test_index, y_label, y_score, y_pred_label, table,
                         table_title, all_samples, fp_samples, "fp", fp_count)
            fp_count += 1
        if y_label[j] == 0 and y_pred_label[j] == 0:  # TN
            write_result(j, test_index, y_label, y_score, y_pred_label, table,
                         table_title, all_samples, tn_samples, "tn", tn_count)
            tn_count += 1
        if y_label[j] == 1 and y_pred_label[j] == 0:  # FN
            write_result(j, test_index, y_label, y_score, y_pred_label, table,
                         table_title, all_samples, fn_samples, "fn", fn_count)
            fn_count += 1
        if y_label[j] == 1 and y_pred_label[j] == 1:  # tp
            write_result(j, test_index, y_label, y_score, y_pred_label, table,
                         table_title, all_samples, tp_samples, "tp", tp_count)
            tp_count += 1
    # write frequency statistic
    # write_frequency(fp_samples,table,table_title,"fp")
    # write_frequency(fn_samples,table,table_title,"fn")
    # write_frequency(tp_samples,table,table_title,"tp")
    # write_frequency(tn_samples,table,table_title,"tn")

    wb.save(file_name + ".xls")


def write_result(j, index, y_label, y_score, y_pred_label, table, table_title, samples_set, samples, group_name, count):
    table.write(j + 1, table_title.index("test_index"), int(index[j]))
    table.write(j + 1, table_title.index("label"), int(y_label[j]))
    table.write(j + 1, table_title.index("prob"), float(y_score[j]))
    table.write(j + 1, table_title.index("pre"), int(y_pred_label[j]))
    samples.extend(samples_set[index[j]])
    table.write(count, table_title.index(group_name), int(index[j]))


def plot_roc(test_labels, test_predictions, table, table_title, file_name):
    fpr, tpr, thresholds = roc_curve(test_labels, test_predictions, pos_label=1)
    threshold = thresholds[np.argmax(tpr - fpr)]
    for i in range(len(fpr)):
        table.write(i + 1, table_title.index("tpr"), tpr[i])
        table.write(i + 1, table_title.index("fpr"), fpr[i])
        table.write(i + 1, table_title.index("thresholds"), float(thresholds[i]))
    table.write(2, table_title.index("threshold"), float(threshold))
    auc = "%.3f" % sklearn.metrics.auc(fpr, tpr)
    title = 'ROC Curve, AUC = ' + str(auc)
    with plt.style.context('ggplot'):
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, "#000099", label='ROC curve')
        ax.plot([0, 1], [0, 1], 'k--', label='Baseline')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.title(title)
        plt.savefig(file_name + '.png', format='png')
        plt.close()
    return threshold


# TODO: 此方法需要重新修改
def imbalance_preprocess(train_dynamic, train_y, name):
    """
    处理数据不平衡问题
    :param train_dynamic:
    :param train_y:
    :return:
    """
    if name == 'LogisticRegression':
        method = SMOTE(kind="regular", random_state=40)
        print(name)
        train_dynamic_res, train_y_res = method.fit_sample(train_dynamic, train_y)
        train_y_res = train_y_res.reshape(-1, 1)
    else:
        method = SMOTE(kind="regular")
        print(name)
        x_res, y_res = method.fit_sample(train_dynamic.reshape([-1, train_dynamic.shape[2]]),
                                         train_y.reshape([-1,train_y.shape[2]]))
        x_size = int(x_res.shape[0]/4)*4
        x_res = x_res[0:x_size,:]
        y_res=y_res[0:x_size]
        train_dynamic_res = x_res.reshape([-1, train_dynamic.shape[1], train_dynamic.shape[2]])
        train_y_res = y_res.reshape([-1, train_y.shape[1], train_y.shape[2]])
    return train_dynamic_res, train_y_res


def imbalance_preprocess(train_dynamic, train_time,train_y, name):
    """
    处理数据不平衡问题
    :param train_dynamic:
    :param train_y:
    :return:
    """
    if name == 'LogisticRegression':
        method = SMOTE(kind="regular", random_state=40)
        print(name)
        train_dynamic_res, train_y_res = method.fit_sample(train_dynamic, train_y)
        train_y_res = train_y_res.reshape(-1, 1)
    else:
        method = SMOTE(kind="regular")
        print(name)
        features = np.concatenate((train_time,train_dynamic),axis=2)
        x_res, y_res = method.fit_sample(features.reshape([-1, features.shape[2]]),
                                         train_y.reshape([-1,train_y.shape[2]]))
        x_size = int(x_res.shape[0]/features.shape[1])*features.shape[1]
        x_res = x_res[0:x_size,:]
        y_res = y_res[0:x_size]
        train_dynamic_res = x_res[:,1:].reshape([-1, train_dynamic.shape[1], train_dynamic.shape[2]])
        train_time_res = x_res[:,0].reshape(-1,train_dynamic.shape[1],1)
        train_y_res = y_res.reshape([-1, train_y.shape[1], train_y.shape[2]])
    return train_dynamic_res,train_time_res, train_y_res


def imbalance_preprocess_gnn(train_features,time,labels):
    all_patient_time = []
    for i in range(time.shape[0]):
        one_patient_time = []
        for j in range(time.shape[1]):
            t = time[i,j,:].reshape(-1,1)
            t_ = np.pad(t,((0, train_features.shape[2]-time.shape[2]),(0, train_features.shape[3] -time.shape[2])), 'constant')
            one_patient_time.append(t_)
        all_patient_time.append(one_patient_time)
    features = np.concatenate((train_features, all_patient_time),axis=3)
    method = SMOTE(kind='regular')
    x_res, y_res = method.fit_sample(features.reshape(-1, features.shape[2]*features.shape[3]),labels.reshape(-1,labels.shape[2]))
    x_size = int(x_res.shape[0]/features.shape[1])*features.shape[1]
    x_res = x_res[0:x_size,:]
    y_res = y_res[0:x_size,]
    train_features_res = x_res.reshape(-1, features.shape[1], features.shape[2], features.shape[3])
    train_x_rex = train_features_res[:,:,:,0:int(train_features_res.shape[3]/2)]
    train_t_res = train_features_res[:,:,0,int(train_features_res.shape[3]/2)].reshape(-1, labels.shape[1],labels.shape[2])
    train_y_res = y_res.reshape(-1, labels.shape[1], labels.shape[2])
    return train_x_rex, train_t_res, train_y_res


class LogisticRegressionExperiment(object):
    def __init__(self):
        self._data_set = read_data("LogisticRegression")
        self._num_features = self._data_set.dynamic_features.shape[1]
        self._time_steps = 1
        self._n_output = 1
        self._model_format()
        self._check_path()

    def _model_format(self):
        learning_rate, max_loss, max_pace, ridge, batch_size, hidden_size, epoch, dropout = lr_setup.all
        self._model = LogisticRegression(num_features=self._num_features,
                                         time_steps=self._time_steps,
                                         n_output=self._n_output,
                                         batch_size=batch_size,
                                         epochs=epoch,
                                         output_n_epoch=ExperimentSetup.output_n_epochs,
                                         learning_rate=learning_rate,
                                         max_loss=max_loss,
                                         dropout=dropout,
                                         max_pace=max_pace,
                                         ridge=ridge)

    def _check_path(self):
        if not os.path.exists("result_9_16_0"):
            os.makedirs("result_9_16_0")
        self._filename = "result_9_16_0" + "/" + self._model.name + " " + \
                         time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    # def do_experiment(self):
    #     train_set_features = self._data_set['train']['x'].reshape(-1,9)
    #     train_set_time = self._data_set['train']['t'].reshape(-1,1)
    #     train_label = self._data_set['train']['e'].reshape(-1,1)
    #     test_features = self._data_set['test']['x'].reshape(-1,9)
    #     test_time = self._data_set['test']['t'].reshape(-1,1)
    #     test_label = self._data_set['test']['e'].reshape(-1,1)
    #     train_dynamic_features = np.concatenate((train_set_time, train_set_features),axis=1)
    #     train_set = DataSet(train_dynamic_features, train_label)
    #     test_dynamic_features = np.concatenate((test_time, test_features),axis=1)
    #     test_set = DataSet(test_dynamic_features, test_label)
    #     self._model.fit(train_set, test_set)
    #     y_score = self._model.predict(test_set)
    #     tol_test_index = np.arange(test_label.shape[0])
    #     evaluate(tol_test_index, test_label, y_score, self._filename)
    #     self._model.close()

    def do_experiments(self):
        n_output = 1
        dynamic_features = self._data_set.dynamic_features
        labels = self._data_set.labels
        # tol_test_index = np.zeros(shape=0, dtype=np.int32)
        tol_pred = np.zeros(shape=(0, n_output))
        tol_label = np.zeros(shape=(0, n_output), dtype=np.int32)
        train_dynamic_features, test_dynamic_features, train_labels, test_labels = \
            split_logistic_data(dynamic_features, labels)
        for i in range(5):
            train_dynamic_res, train_labels_res = imbalance_preprocess(train_dynamic_features[i], train_labels[i],
                                                                       'LogisticRegression')
            train_set = DataSet(train_dynamic_res, train_labels_res)
            test_set = DataSet(test_dynamic_features[i].reshape(-1, 93), test_labels[i].reshape(-1, 1))
            self._model.fit(train_set, test_set)
            y_score = self._model.predict(test_set)
            tol_pred = np.vstack((tol_pred, y_score))
            tol_label = np.vstack((tol_label, test_labels[i]))
            print("Cross validation: {} of {}".format(i, 5),
                  time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        tol_test_index = np.arange(labels.shape[0] * labels.shape[1])
        evaluate(tol_test_index, tol_label, tol_pred, self._filename)
        self._model.close()


class BidirectionalLSTMExperiments(object):
    def __init__(self):
        # self._data_set = get_pick_data("BidirectionalLSTM")
        self._data_set = read_data("rnn")
        # self._num_features = self._data_set.dynamic_features.shape[2]-1
        # self._num_features = (self._data_set.dynamic_features.shape[2])*self._data_set.dynamic_features.shape[3]
        self._time_steps = self._data_set.dynamic_features.shape[1]
        self._num_features = self._data_set.dynamic_features.shape[2]
        self._time_steps = self._data_set.dynamic_features.shape[1]
        self._n_output = 1
        self._model_format()
        self._check_path()

    def _model_format(self):
        learning_rate, max_loss, max_pace, ridge, batch_size, hidden_size, epochs, dropout = bi_lstm_setup.all
        self._model = BidirectionalLSTMModel(time_steps=self._time_steps,
                                             num_features=self._num_features,
                                             lstm_size=hidden_size,
                                             n_output=self._n_output,
                                             batch_size=batch_size,
                                             epochs=epochs,
                                             output_n_epoch=ExperimentSetup.output_n_epochs,
                                             learning_rate=learning_rate,
                                             max_loss=max_loss,
                                             dropout=dropout,
                                             max_pace=max_pace,
                                             ridge=ridge)

    def _check_path(self):
        if not os.path.exists("result_9_16_0"):
            os.makedirs("result_9_16_0")
        self._filename = "result_9_16_0" + "/" + self._model.name + " " + time.strftime("%Y-%m-%d-%H-%M-%S",
                                                                                        time.localtime())

    # def do_experiment(self):
    #     train_set_features = self._data_set['train']['x'].reshape(-1,1,9)
    #     train_set_time = self._data_set['train']['t'].reshape(-1,1,1)
    #     train_label = self._data_set['train']['e'].reshape(-1,1,1)
    #     test_features = self._data_set['test']['x'].reshape(-1,1,9)
    #     test_time = self._data_set['test']['t'].reshape(-1,1,1)
    #     test_label = self._data_set['test']['e'].reshape(-1,1,1)
    #     train_dynamic_features = np.concatenate((train_set_time, train_set_features),axis=2)
    #     train_set = DataSet(train_dynamic_features, train_label)
    #     test_dynamic_features = np.concatenate((test_time, test_features),axis=2)
    #     test_set = DataSet(test_dynamic_features, test_label)
    #     self._model.fit(train_set, test_set)
    #     y_score = self._model.predict(test_set)
    #     tol_test_index = np.arange(test_label.shape[0])
    #     evaluate(tol_test_index, test_label, y_score, self._filename)
    #     self._model.close()

    def do_experiments(self):
        n_output = 1
        dynamic_features = self._data_set.dynamic_features
        labels = self._data_set.labels
        time = self._data_set.time
        # tol_test_index = np.zeros(shape=0, dtype=np.int32)
        tol_pred = np.zeros(shape=(0, dynamic_features.shape[1], n_output))
        tol_label = np.zeros(shape=(0, dynamic_features.shape[1], n_output), dtype=np.int32)
        # train_dynamic_features, test_dynamic_features, train_time, test_time, train_labels, test_labels = split_data_set_gnn(dynamic_features,time,
        #                                                                                           labels)
        train_dynamic_features, test_dynamic_features, train_time, test_time, train_labels, test_labels = split_data_set(
            dynamic_features, time,labels)
        for i in range(5):
            train_dynamic_res, train_time_res,train_labels_res = imbalance_preprocess(train_dynamic_features[i], train_time[i],train_labels[i],"rnn")
            # train_dynamic_res, train_time_res, train_labels_res = imbalance_preprocess(train_dynamic_features[i],
            #                                                                                train_time[i],
            #                                                                                train_labels[i])
            train_set = DataSet(train_dynamic_res, train_time_res,train_labels_res)
            # train_set = DataSet(train_dynamic_features[i],train_labels[i])
            test_set = DataSet(test_dynamic_features[i],test_time[i],test_labels[i])
            self._model.fit(train_set, test_set)
            y_score = self._model.predict(test_set)
            tol_pred = np.vstack((tol_pred, y_score))
            tol_label = np.vstack((tol_label, test_labels[i]))
            print("cross validation  " + str(i) + "  have finished")
        tol_test_index = np.arange(labels.shape[0] * labels.shape[1])
        evaluate(tol_test_index, tol_label, tol_pred, self._filename)
        self._model.close()


class GNNlstmSurvExperiment(object):
    def __init__(self):
        self._data_set = read_data("rnn")
        # self._data_set = read_WHAS_dataset()
        # self._num_features = 9
        # self._time_steps = 1
        # self._feature_dims = 1
        self._num_features = self._data_set.dynamic_features.shape[2]
        self._time_steps = self._data_set.dynamic_features.shape[1]
        self._feature_dims = self._data_set.dynamic_features.shape[3]
        self._n_output = 1
        self.model_format()
        self.check_path()

    def model_format(self):
        learning_rate,max_loss,max_pace,ridge,batch_size,hidden_size,epoch,dropout = gnn_lstm_setup.all
        self._model = GnnLSTMSurV(learning_rate=learning_rate,
                                  max_loss=max_loss,
                                  max_pace=max_pace,
                                  ridge=ridge,
                                  batch_size=batch_size,
                                  hidden_size=hidden_size,
                                  epoch=epoch,
                                  dropout=dropout,
                                  n_output=ExperimentSetup.output_n_epochs,
                                  num_features=self._num_features,
                                  feature_dims=self._feature_dims,
                                  time_steps=self._time_steps)

    def check_path(self):
        if not os.path.exists("result_gnn"):
            os.mkdir("result_gnn")
        self._filename = "result_gnn"+"/" + self._model._name + " " + time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime())

    def do_experiments(self):
        n_output = 1
        dynamic_features = self._data_set.dynamic_features
        time = self._data_set.time
        labels = self._data_set.labels
        tol_pred = np.zeros(shape=(0))
        tol_labels = np.zeros(shape=(0))
        train_dynamic_features,test_dynamic_features,train_time, test_time,train_labels,test_labels = \
            split_data_set_gnn(dynamic_features,time,labels)
        for i in range(5):
            train_x_res,train_t_res,train_labels_res = imbalance_preprocess_gnn(train_dynamic_features[i],train_time[i],train_labels[i])
            train_set = DataSet(train_x_res,train_t_res,train_labels_res)
            # train_set = DataSet(train_dynamic_features[i],train_time[i],train_labels[i])
            test_set = DataSet(test_dynamic_features[i],test_time[i],test_labels[i])
            self._model.fit(train_set,test_set)
            y_score, test_y = self._model.predict(test_set)
            y_score = np.array(y_score)
            test_y = np.array(test_y)
            test_y_score = np.zeros(shape=(0))
            test_y_all = np.zeros(shape=(0))
            for m in range(y_score.shape[0]):
                test_y_score = np.concatenate((test_y_score,y_score[m].reshape(-1,)))
                test_y_all = np.concatenate((test_y_all,test_y[m].reshape(-1,)))
            tol_labels = np.concatenate((tol_labels,test_y_all.reshape(-1,)))
            tol_pred = np.concatenate((tol_pred,test_y_score.reshape(-1)))
            print("cross validation" + str(i) + "have finished")
        tol_test_index = np.arange(tol_labels.shape[0])
        evaluate(tol_test_index, tol_labels, tol_pred, self._filename)
        self._model.close()

    def attention_analysis(self):
        attention_signals_tol = np.zeros(shape=(0, self._num_features,self._feature_dims))
        graph_tol = np.zeros(shape=(0,self._time_steps,self._num_features,self._num_features))
        models = ["save_net12-28-17-05-44.ckpt", "save_net12-28-17-06-58.ckpt",
                  "save_net12-28-17-08-13.ckpt", "save_net12-28-17-09-27.ckpt",
                  "save_net12-28-17-10-42.ckpt"]
        # n_output = 1
        dynamic_features = self._data_set.dynamic_features
        time = self._data_set.time
        labels = self._data_set.labels
        train_dynamic_features, test_dynamic_features, train_time, test_time, train_labels, test_labels = split_data_set_gnn(
            dynamic_features,
            time, labels)
        for j in range(5):
            test_set = DataSet(test_dynamic_features[j],test_time[j],test_labels[j])
            prob, attention_weight,graph = self._model.attention_analysis(test_set.dynamic_features, test_set.time,models[j])
            graph = graph.reshape(-1,self._time_steps,self._num_features,self._num_features)
            attention_signals_tol = np.concatenate((attention_signals_tol, attention_weight))
            graph_tol = np.concatenate((graph_tol,graph))
        np.save("GNN_Attention_5_.npy", attention_signals_tol)
        np.save("Adjacency_weight_5_.npy",graph_tol)


    def do_experiment(self):
        train_set_features = self._data_set['train']['x'].reshape(-1,1,9,1)
        train_set_time = self._data_set['train']['t'].reshape(-1,1,1)
        train_label = self._data_set['train']['e'].reshape(-1,1,1)
        test_features = self._data_set['test']['x'].reshape(-1,1,9,1)
        test_time = self._data_set['test']['t'].reshape(-1,1,1)
        test_label = self._data_set['test']['e'].reshape(-1,1,1)
        train_set = DataSet(train_set_features,train_set_time,train_label)
        test_set = DataSet(test_features,test_time,test_label)
        self._model.fit(train_set, test_set)
        test_y_score = np.zeros(shape=[0])
        test_y_all = np.zeros(shape=[0])
        y_score,y_label = self._model.predict(test_set)
        y_score = np.array(y_score)
        for i in range(y_score.shape[0]):
            test_y_score = np.concatenate((test_y_score,y_score[i].reshape(-1,)))
            test_y_all = np.concatenate((test_y_all,y_label[i].reshape(-1,)))
        tol_test_index = np.arange(test_label.shape[0])
        evaluate(tol_test_index, test_y_all, test_y_score, self._filename)
        self._model.close()


class AttentionBiLSTMExperiments(BidirectionalLSTMExperiments):
    def __init__(self):
        super().__init__()

    def _model_format(self):
        learning_rate, max_loss, max_pace, ridge, batch_size, hidden_size, epochs, dropout = global_rnn_setup.all
        self._model = AttentionLSTMModel(num_features=self._num_features,
                                         time_steps=self._time_steps,
                                         lstm_size=hidden_size,
                                         n_output=self._n_output,
                                         batch_size=batch_size,
                                         epochs=epochs,
                                         output_n_epoch=ExperimentSetup.output_n_epochs,
                                         learning_rate=learning_rate,
                                         max_loss=max_loss,
                                         max_pace=max_pace,
                                         dropout=dropout,
                                         ridge=ridge)

    # 得到prediction中的attention weight
    def attention_analysis(self):
        attention_signals_tol = np.zeros(shape=(0, self._time_steps, self._num_features))
        models = ["save_net10-23-21-17-49.ckpt", "save_net10-23-21-18-30.ckpt",
                  "save_net10-23-21-19-11.ckpt", "save_net10-23-21-19-53.ckpt",
                  "save_net10-23-21-20-34.ckpt"]
        # n_output = 1
        dynamic_features = self._data_set.dynamic_features
        time = self._data_set.time
        labels = self._data_set.labels
        train_dynamic_features, test_dynamic_features, train_time, test_time,train_labels, test_labels = split_data_set(dynamic_features,
                                                                                                  time,labels)
        for j in range(5):
            test_set = DataSet(test_dynamic_features[j], test_labels[j])
            prob, attention_weight = self._model.attention_analysis(test_set.dynamic_features, models[j])
            attention_signals_tol = np.concatenate((attention_signals_tol, attention_weight))
        np.save("allAttentionWeight_5.npy", attention_signals_tol)

    # 得到每一个特征的类别
    def cluster_by_attention_weight(self):
        attention_weight = np.load("average_weight.npy")
        attention_weight_array = attention_weight.reshape([-1, self._num_features])
        all_feature_breaks = []
        for nums in range(self._num_features):
            one_feature_breaks = jenkspy.jenks_breaks(attention_weight_array[:, nums], nb_class=5)
            print(one_feature_breaks)
            all_feature_breaks.append(one_feature_breaks)
        np.save("all_features_breaks_ave.npy", all_feature_breaks)

    # 得到每次住院记录的stage
    @staticmethod
    def get_stages():
        all_features_breaks = np.load("all_features_breaks_ave.npy")
        attention_weight = np.load("average_weight.npy")
        one_patient_stage = []
        one_patient_score = []
        all_patient_stage = np.zeros(shape=(0, 5), dtype=np.int32)
        all_patient_score = np.zeros(shape=(0, 5), dtype=np.float32)
        for patient in range(attention_weight.shape[0]):
            for visit in range(attention_weight.shape[1]):
                one_patient_features = attention_weight[patient, visit, :]
                patient_features_in_stage1 = []
                patient_features_in_stage2 = []
                patient_features_in_stage3 = []
                patient_features_in_stage4 = []
                patient_features_in_stage5 = []
                for i in range(one_patient_features.shape[0]):
                    if one_patient_features[i] < all_features_breaks[i, 1]:
                        patient_features_in_stage1.append(one_patient_features[i])

                    if all_features_breaks[i, 1] <= one_patient_features[i] < all_features_breaks[i, 2]:
                        patient_features_in_stage2.append(one_patient_features[i])

                    if all_features_breaks[i, 2] <= one_patient_features[i] < all_features_breaks[i, 3]:
                        patient_features_in_stage3.append(one_patient_features[i])

                    if all_features_breaks[i, 3] <= one_patient_features[i] <= all_features_breaks[i, 4]:
                        patient_features_in_stage4.append(one_patient_features[i])

                    # if all_features_breaks[i,4] <= one_patient_features[i] <= all_features_breaks[i,5]:
                    #     patient_features_in_stage5.append(one_patient_features[i])

                stage1_score = np.sum(patient_features_in_stage1)
                stage2_score = np.sum(patient_features_in_stage2)
                stage3_score = np.sum(patient_features_in_stage3)
                stage4_score = np.sum(patient_features_in_stage4)
                stage5_score = np.sum(patient_features_in_stage5)
                # score = [stage1_score,stage2_score,stage3_score,stage4_score,stage5_score]
                score = [stage1_score, stage2_score, stage3_score, stage4_score]
                max_score = max(score)
                max_score_index = np.argmax(score)
                one_patient_score.append(max_score)
                one_patient_stage.append(max_score_index)
            all_patient_stage = np.concatenate((all_patient_stage, np.array(one_patient_stage).reshape(-1, 5)))
            all_patient_score = np.concatenate((all_patient_score, np.array(one_patient_score).reshape(-1, 5)))
            one_patient_stage = []
            one_patient_score = []
        np.save('all_patient_stage_ave.npy', all_patient_stage)
        np.save('all_patient_score_ave.npy', all_patient_score)

    # 采用k-means cluster 得到每个feature的label
    def k_means_weight_stages(self):
        all_patient_stage = np.zeros(shape=(10500, 0), dtype=np.int32)
        km = KMeans(n_clusters=3)
        attention_weight = np.load("average_weight.npy")
        attention_weight = attention_weight.reshape(-1, self._num_features)
        for nums in range(self._num_features):
            data = attention_weight[:, nums].reshape(-1, 1)
            km.fit(data)
            labels = km.labels_.reshape(-1, 1)
            all_patient_stage = np.concatenate((all_patient_stage, labels), axis=1)
        print(all_patient_stage)
        all_patient_stage = all_patient_stage.reshape(-1, 5, 92)
        np.save("k_means_stages_3.npy", all_patient_stage)
        return all_patient_stage

    # k-means 结果
    @staticmethod
    def get_kmeans_stages():
        attention_weight = np.load("average_weight.npy")
        attention_stage = np.load("kmeans_stages_4.npy")
        labels = np.load("pick_5_visit_labels_merge_1.npy")[0:2100, :, -1].reshape(-1, 5, 1)
        all_patient_stage = np.zeros(shape=(0, 5), dtype=np.int32)
        patient_in_stage0 = np.zeros(shape=(0, 92), dtype=np.int32)
        patient_in_stage1 = np.zeros(shape=(0, 92), dtype=np.int32)
        patient_in_stage2 = np.zeros(shape=(0, 92), dtype=np.int32)
        patient_in_stage3 = np.zeros(shape=(0, 92), dtype=np.int32)
        patient_in_stage4 = np.zeros(shape=(0, 92), dtype=np.int32)

        patient_in_stage0_labels = np.zeros(shape=(0, 1), dtype=np.int32)
        patient_in_stage1_labels = np.zeros(shape=(0, 1), dtype=np.int32)
        patient_in_stage2_labels = np.zeros(shape=(0, 1), dtype=np.int32)
        patient_in_stage3_labels = np.zeros(shape=(0, 1), dtype=np.int32)
        patient_in_stage4_labels = np.zeros(shape=(0, 1), dtype=np.int32)
        for patient in range(attention_weight.shape[0]):
            one_patient_stage = []
            for visit in range(attention_weight.shape[1]):
                one_patient_weight = attention_weight[patient, visit, :].reshape(-1, 92)
                label = labels[patient, visit, :].reshape(-1, 1)
                one_patient_weight_stage = attention_stage[patient, visit, :].reshape(-1, 92)
                weight_in_stage0 = np.sum(one_patient_weight[np.where(one_patient_weight_stage == 0)])
                weight_in_stage1 = np.sum(one_patient_weight[np.where(one_patient_weight_stage == 1)])
                weight_in_stage2 = np.sum(one_patient_weight[np.where(one_patient_weight_stage == 2)])
                weight_in_stage3 = np.sum(one_patient_weight[np.where(one_patient_weight_stage == 3)])
                weight_in_stage4 = np.sum(one_patient_weight[np.where(one_patient_weight_stage == 4)])
                one_visit_score = np.max([weight_in_stage0, weight_in_stage1,
                                          weight_in_stage2, weight_in_stage3, weight_in_stage4])
                one_visit_stage = np.argmax([weight_in_stage0, weight_in_stage1,
                                             weight_in_stage2, weight_in_stage3, weight_in_stage4])
                one_patient_stage.append(one_visit_stage)
                if one_visit_stage == 0:
                    patient_in_stage0 = np.concatenate((patient_in_stage0, one_patient_weight))
                    patient_in_stage0_labels = np.concatenate((patient_in_stage0_labels, label))
                if one_visit_stage == 1:
                    patient_in_stage1 = np.concatenate((patient_in_stage1, one_patient_weight))
                    patient_in_stage1_labels = np.concatenate((patient_in_stage1_labels, label))
                if one_visit_stage == 2:
                    patient_in_stage2 = np.concatenate((patient_in_stage2, one_patient_weight))
                    patient_in_stage2_labels = np.concatenate((patient_in_stage2_labels, label))
                if one_visit_stage == 3:
                    patient_in_stage3 = np.concatenate((patient_in_stage3, one_patient_weight))
                    patient_in_stage3_labels = np.concatenate((patient_in_stage3_labels, label))
                if one_visit_stage == 4:
                    patient_in_stage4 = np.concatenate((patient_in_stage4, one_patient_weight))
                    patient_in_stage4_labels = np.concatenate((patient_in_stage4_labels, label))
            all_patient_stage = np.concatenate((all_patient_stage, np.array(one_patient_stage).reshape(-1, 5)))
        print(patient_in_stage0.shape[0])
        print(patient_in_stage1.shape[0])
        print(patient_in_stage2.shape[0])
        print(patient_in_stage3.shape[0])
        print(patient_in_stage4.shape[0])

        death_rate = {}
        death_rate["stage_0"] = len(np.where(patient_in_stage0_labels == 1)[0]) / patient_in_stage0_labels.shape[0]
        death_rate["stage_1"] = len(np.where(patient_in_stage1_labels == 1)[0]) / patient_in_stage1_labels.shape[0]
        death_rate["stage_2"] = len(np.where(patient_in_stage2_labels == 1)[0]) / patient_in_stage2_labels.shape[0]
        # death_rate["stage_3"] = len(np.where(patient_in_stage3_labels == 1)[0]) / patient_in_stage3_labels.shape[0]
        # death_rate["stage_4"] = len(np.where(patient_in_stage4_labels == 1)[0]) / patient_in_stage4_labels.shape[0]

        stage_0_mean = np.mean(patient_in_stage0, axis=0).reshape(-1)
        stage_1_mean = np.mean(patient_in_stage1, axis=0).reshape(-1)
        stage_2_mean = np.mean(patient_in_stage2, axis=0)
        stage_3_mean = np.mean(patient_in_stage3, axis=0)
        stage_4_mean = np.mean(patient_in_stage4, axis=0)
        data_frame = pd.DataFrame({"stage0": stage_0_mean, "stage1": stage_1_mean, "stage2": stage_2_mean,
                                   "stage3": stage_3_mean, "stage4": stage_4_mean})
        data_frame.to_csv("stage_weights_1.csv", index=False, sep=",")

        stage_0_max_index = np.argsort(stage_0_mean)
        stage_1_max_index = np.argsort(stage_1_mean)
        stage_2_max_index = np.argsort(stage_2_mean)
        stage_3_max_index = np.argsort(stage_3_mean)
        stage_4_max_index = np.argsort(stage_4_mean)
        dataframe2 = pd.DataFrame(
            {"stage0": stage_0_max_index, "stage1": stage_1_max_index, "stage2": stage_2_max_index,
             "stage3": stage_3_max_index, "stage4": stage_4_max_index})
        dataframe2.to_csv("前十的weight.csv", index=False, sep=",")

        print(stage_0_max_index)


class SelfAttentionBiLSTMExperiments(BidirectionalLSTMExperiments):
    def __init__(self):
        super().__init__()

    def _model_format(self):
        learning_rate, max_loss, max_pace, ridge, batch_size, hidden_size, epochs, dropout = self_rnn_setup.all
        self._model = SelfAttentionLSTMModel(num_features=self._num_features,
                                             time_steps=self._time_steps,
                                             lstm_size=hidden_size,
                                             n_output=self._n_output,
                                             batch_size=batch_size,
                                             epochs=epochs,
                                             output_n_epoch=ExperimentSetup.output_n_epochs,
                                             learning_rate=learning_rate,
                                             max_loss=max_loss,
                                             max_pace=max_pace,
                                             dropout=dropout,
                                             ridge=ridge)


#  (数据需要重新整理成不相关的独立变量 所以应该使用没有二值化的数据)cox regression model(已将完成)
def cox_regression_experiment():
    dynamic_features = np.load('pick_5_visit_features_merge_1.npy')[0:2100, :, :-2]
    dynamic_features.astype(np.int32)
    labels = np.load('pick_5_visit_labels_merge_1.npy')[:, :, -4].reshape(-1, dynamic_features.shape[1], 1)
    data = np.concatenate((dynamic_features, labels), axis=2).reshape(-1, 94)
    data_set = pd.DataFrame(data)
    col_list = list(data_set.columns.values)
    new_col = [str(x) for x in col_list]
    data_set.columns = new_col
    np.savetxt('allPatient_now.csv', data_set, delimiter=',')
    print(list(data_set.columns.values))
    cph = CoxPHFitter(penalizer=100)
    cph.fit(data_set, duration_col='0', event_col='93', show_progress=True)
    cph.print_summary()
    # cph.plot(columns=['15','20','21','25'])
    # plt.savefig('cox model' + '.png', format='png')

    scores = k_fold_cross_validation(cph, data_set, '0', event_col='93', k=5)
    print(scores)
    print(np.mean(scores))
    print(np.std(scores))


def rsf_experiment():
    time_line = range(0, 4000, 1)
    rsf = RandomSurvivalForest(n_estimators=10, timeline=time_line)
    dynamic_features = np.load("pick_5_visit_features_merge_1.npy")[0:2100, :, 1:93]
    time = np.load("pick_5_visit_features_merge_1.npy")[0:2100, :, 0].reshape(-1, 5, 1)
    event = np.load("pick_5_visit_labels_merge_1.npy")[0:2100, :, -1].reshape(-1, 5, 1)
    labels = np.concatenate((time, event), axis=2)
    train_features, test_features, train_labels, test_labels = split_data_set(dynamic_features, labels)
    c_index = {}

    for j in range(5):
        rsf.fit(pd.DataFrame(train_features[j].reshape(-1, 92)), pd.DataFrame(train_labels[j].reshape(-1, 2)))
        train_c_index = rsf.oob_score
        print("train_c_index:{:.2f}".format(train_c_index))
        y_pred = rsf.predict(test_features[j].reshape(-1, 92))
        c_index[j] = concordance_index(test_labels[j][:, :, 0].reshape(-1), y_pred, test_labels[j][:, :, 1].reshape(-1))
        print("c_index:{:.2f}".format(c_index[j]))
    print(c_index)


def get_average_weight():
    weight1 = np.load("GNN_Attention_1_.npy")
    weight1_ = (weight1[:,:,0]+weight1[:,:,1])/2
    weight2 = np.load("GNN_Attention_2_.npy")
    weight2_ = (weight2[:, :, 0] + weight2[:, :, 1]) / 2
    weight3 = np.load("GNN_Attention_3_.npy")
    weight3_ = (weight3[:, :, 0] + weight3[:, :, 1]) / 2
    weight4 = np.load("GNN_Attention_4_.npy")
    weight4_ = (weight4[:, :, 0] + weight4[:, :, 1]) / 2
    weight5 = np.load("GNN_Attention_5_.npy")
    weight5_ = (weight5[:, :, 0] + weight5[:, :, 1]) / 2
    ave_weight = np.add(weight1_, np.add(weight2_, np.add(weight3_, np.add(weight4_, weight5_)))) / 5.0
    print(ave_weight)
    np.save("average_weight_.npy", ave_weight)
    feature_weight_new = ave_weight.reshape(-1, ave_weight.shape[1])
    weight = np.mean(feature_weight_new, axis=0)
    np.savetxt("feature_selection_.csv", weight, delimiter=',')
    return ave_weight


def get_average_adjacency():
    adjacency_1 = np.load("Adjacency_weight_1_.npy")
    adjacency_2 = np.load("Adjacency_weight_2_.npy")
    adjacency_3 = np.load("Adjacency_weight_3_.npy")
    adjacency_4 = np.load("Adjacency_weight_4_.npy")
    adjacency_5 = np.load("Adjacency_weight_5_.npy")
    adjacency_ave = (adjacency_1+adjacency_2+adjacency_3+adjacency_4+adjacency_5)/5.0
    adjacency_ave = adjacency_ave.reshape(-1,adjacency_ave.shape[2],adjacency_ave.shape[3])
    adjacency_final = np.mean(adjacency_ave,axis=0)
    np.save("adjacency_ave_.npy",adjacency_final)
    np.savetxt("adjacency_ave_.csv", adjacency_final, delimiter=',')

def evaluate_model(model, dataset, bootstrap=False):
    def ci(model):
        def cph_ci(x, t, e, **kwargs):
            return concordance_index(
                event_times=t,
                predicted_scores=-model.predict_partial_hazard(x),
                event_observed=e,
            )

        return cph_ci

    def mse(model):
        def cph_mse(x, hr, **kwargs):
            hr_pred = np.squeeze(-model.predict_partial_hazard(x).values)
            return ((hr_pred - hr) ** 2).mean()

        return cph_mse

    metrics = {}

    # Calculate c_index
    metrics['c_index'] = ci(model)(**dataset)
    return metrics


# 得到的结果不一致？（按照linux上的包来写）
def compare_cox_regression():
    data_sets = read_data()
    x_train = np.array(data_sets['train']['x']).reshape(-1,9)
    t_train = np.array(data_sets['train']['t']).reshape(-1,1)
    e_train = np.array(data_sets['train']['e']).reshape(-1,1)
    train = np.concatenate((x_train, np.concatenate((t_train,e_train),axis=1)),axis=1)
    train_df = pd.DataFrame(train)
    new_col = [str(x) for x in range(train.shape[1])]
    train_df.columns = new_col

    cph = CoxPHFitter()
    cph.fit(train_df,duration_col='6',event_col='7', show_progress=True)
    cph.print_summary()
    metrics = evaluate_model(cph,data_sets['test'])
    print("Test metrics: " + str(metrics))


if __name__ == "__main__":
    get_average_adjacency()
    get_average_weight()
    for i in range(5):
        # BidirectionalLSTMExperiments().do_experiments()
        GNNlstmSurvExperiment().attention_analysis()
        # LogisticRegressionExperiment().do_experiments()
        # BidirectionalLSTMExperiments().do_experiments()
        # AttentionBiLSTMExperiments().do_experiments()
        # AttentionBiLSTMExperiments().get_stages()
        # SelfAttentionBiLSTMExperiments().do_experiments()
