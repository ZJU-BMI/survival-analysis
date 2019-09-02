import os
from collections import Counter
import matplotlib.pyplot as plt
import sklearn
import time
import xlwt
import xlsxwriter
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score,f1_score,precision_score,recall_score,roc_curve
from data import read_data,DataSet
from models import BidirectionalLSTMModel,AttentionLSTMModel,LogisticRegression


class ExperimentSetup(object):
    kfold = 5  # 5折交叉验证
    batch_size = 128
    hidden_size = 512
    epochs = 20
    output_n_epochs = 1

    def __init__(self,learning_rate, max_loss=2.0, max_pace=0.01,ridge=0.0):
        self._learning_rate = learning_rate
        self._max_loss = max_loss
        self._max_pace = max_pace
        self._ridge = ridge

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
    def all(self):
        return self._learning_rate, self._max_loss, self._max_pace, self._ridge


# set the parameters
lr_steup = ExperimentSetup(0.01,2,0.0001,0.0001)
bi_lstm_setup = ExperimentSetup(0.05,0.5,0.01,0.001)
ca_rnn_seup = ExperimentSetup(0.0001,0.08,0.001,0.001)


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
    y_label = y_label.reshape([-1,1])
    y_score= y_score.reshape([-1,1])
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
    all_samples = np.load("logistic_features.npy")
    for j in range(len(y_label)):
        if y_label[j] ==0 and y_pred_label[j] ==1:  # FP
            write_result(j,test_index,y_label,y_score,y_pred_label,table,table_title,all_samples,fp_samples,"fp",fp_count)
            fp_count += 1
        if y_label[j] ==0 and y_pred_label[j] ==0: # TN
            write_result(j,test_index,y_label,y_score,y_pred_label,table,table_title,all_samples,tn_samples,"tn",tn_count)
            tn_count += 1
        if y_label[j] ==1 and y_pred_label[j]==0:   # FN
            write_result(j,test_index,y_label,y_score,y_pred_label,table,table_title,all_samples,fn_samples,"fn",fn_count)
            fn_count += 1
        if y_label[j] ==1 and y_pred_label[j] ==1:  # tp
            write_result(j,test_index,y_label,y_score,y_pred_label,table,table_title,all_samples,tp_samples,"tp",tp_count)
            tp_count += 1
    # write frequency statistic
    # write_frequency(fp_samples,table,table_title,"fp")
    # write_frequency(fn_samples,table,table_title,"fn")
    # write_frequency(tp_samples,table,table_title,"tp")
    # write_frequency(tn_samples,table,table_title,"tn")

    wb.save(file_name + ".xls")


def write_result(j, index, y_label,y_score,y_pred_label, table, table_title, samples_set, samples, group_name, count):
    table.write(j+1, table_title.index("test_index"), int(index[j]))
    table.write(j+1, table_title.index("label"), int(y_label[j]))
    table.write(j+1, table_title.index("prob"),float(y_score[j]))
    table.write(j+1, table_title.index("pre"),int(y_pred_label[j]))
    samples.extend(samples_set[index[j]])
    table.write(count, table_title.index(group_name),int(index[j]))


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
        method = SMOTE(kind="regular",random_state=40)
        print(name)
        train_dynamic_res, train_y_res = method.fit_sample(train_dynamic, train_y)
    else:
        method = SMOTE(kind="regular")
        print(name)
        x_res, y_res = method.fit_sample(train_dynamic.reshape([-1, train_dynamic.shape[2]]),
                                         train_y.reshape([-1,train_y.shape[2]]))
        x_res = x_res[0:11915,:]
        y_res=y_res[0:11915]
        train_dynamic_res = x_res.reshape([-1, train_dynamic.shape[1], train_dynamic.shape[2]])
        train_y_res = y_res.reshape([-1,train_y.shape[1], train_y.shape[2]])
    return train_dynamic_res, train_y_res


class LogisticRegressionExperiment(object):
    def __init__(self):
        self._data_set = read_data("LogisticRegression")
        self._num_features = self._data_set.dynamic_features.shape[1]
        self._time_steps = 1
        self._n_output = 1
        self._model_format()
        self._check_path()

    def _model_format(self):
        learning_rate, max_loss, max_pace, ridge = lr_steup.all
        self._model = LogisticRegression(num_features=self._num_features,
                                         time_steps = self._time_steps,
                                         n_output = self._n_output,
                                         batch_size = ExperimentSetup.batch_size,
                                         epochs = ExperimentSetup.epochs,
                                         output_n_epoch = ExperimentSetup.output_n_epochs,
                                         learning_rate = learning_rate,
                                         max_loss = max_loss,
                                         max_pace = max_pace,
                                         ridge = ridge)

    def _check_path(self):
        if not os.path.exists("average_result_test"):
            os.makedirs("average_result_test")
        self._filename = "average_result_test" + "/" + self._model.name + " " + time.strftime( "%Y-%m-%d-%H-%M-%S", time.localtime())

    def do_experiments(self):
        dynamic_features = self._data_set.dynamic_features
        labels = self._data_set.labels
        labels = labels.astype('int')
        kf = sklearn.model_selection.StratifiedKFold(n_splits=ExperimentSetup.kfold, shuffle=False)
        n_output = 1
        tol_test_index = np.zeros(shape=0, dtype=np.int32)
        tol_pred = np.zeros(shape=(0,n_output))
        tol_label = np.zeros(shape=(0,n_output),dtype=np.int32)
        i = 1
        for train_index, test_index in kf.split(X= dynamic_features, y=labels):
            train_dynamic = dynamic_features[train_index]
            train_y= labels[train_index]
            train_dynamic_res, train_y_res = imbalance_preprocess(train_dynamic, train_y, 'LogisticRegression')
            test_dynamic = dynamic_features[test_index]
            test_y = labels[test_index]
            train_set = DataSet(train_dynamic_res, train_y_res.reshape([-1,1]))
            test_set = DataSet(test_dynamic, test_y)
            self._model.fit(train_set,test_set)
            y_score = self._model.predict(test_set)
            tol_test_index = np.concatenate((tol_test_index, test_index))
            tol_pred = np.vstack((tol_pred, y_score))
            tol_label = np.vstack((tol_label, test_y))
            print("Cross validation: {} of {}".format(i, ExperimentSetup.kfold),
                  time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            i += 1
        evaluate(tol_test_index, tol_label, tol_pred, self._filename)
        self._model.close()


class BidirectionalLSTMExperiments(object):
    def __init__(self):
        self._data_set = read_data("BidirectionalLSTM")
        self._num_features = self._data_set.dynamic_features.shape[2]
        self._time_steps = self._data_set.dynamic_features.shape[1]
        self._n_output = 1
        self._model_format()
        self._check_path()

    def _model_format(self):
        learning_rate, max_loss, max_pace, ridge =bi_lstm_setup.all
        self._model = BidirectionalLSTMModel(time_steps=self._time_steps,
                                             num_features=self._num_features,
                                             lstm_size=ExperimentSetup.hidden_size,
                                             n_output=self._n_output,
                                             batch_size=ExperimentSetup.batch_size,
                                             epochs=ExperimentSetup.epochs,
                                             output_n_epoch=ExperimentSetup.output_n_epochs,
                                             learning_rate=learning_rate,
                                             max_loss=max_loss,
                                             max_pace=max_pace,
                                             ridge=ridge)

    def _check_path(self):
        if not os.path.exists("average_result_test"):
            os.makedirs("average_result_test")
        self._filename = "average_result_test" + "/" + self._model.name + " " + time.strftime( "%Y-%m-%d-%H-%M-%S", time.localtime())

    def do_experiments(self):
        for i in range(5):
            dynamic_features = self._data_set.dynamic_features
            labels = self._data_set.labels
            labels = labels.astype('int')
            x_train,x_test,y_train,y_test = train_test_split(dynamic_features,labels,test_size=0.4,random_state=1)
            train_dynamic_res, train_y_res = imbalance_preprocess(x_train, y_train, 'BiLSTM')
            train_set = DataSet(train_dynamic_res,train_y_res)
            test_set = DataSet(x_test,y_test)
            self._model.fit(train_set, test_set)
            y_score = self._model.predict(test_set)
            test_index = np.arange(y_test.shape[0]*y_test.shape[1])
            evaluate(test_index, y_test, y_score, self._filename)
        self._model.close()


class AttentionBiLSTMExperiments(BidirectionalLSTMExperiments):
    def __init__(self):
        super().__init__()

    def _model_format(self):
        learning_rate, max_loss, max_pace, ridge = ca_rnn_seup.all
        self._model = AttentionLSTMModel(num_features=self._num_features,
                                         time_steps=self._time_steps,
                                         lstm_size=ExperimentSetup.hidden_size,
                                         n_output=self._n_output,
                                         batch_size=ExperimentSetup.batch_size,
                                         epochs=ExperimentSetup.epochs,
                                         output_n_epoch=ExperimentSetup.output_n_epochs,
                                         learning_rate=learning_rate,
                                         max_loss=max_loss,
                                         max_pace=max_pace,
                                         ridge=ridge)



if __name__ == "__main__":
    for i in range(5):
        # LogisticRegressionExperiment().do_experiments()
        # BidirectionalLSTMExperiments().do_experiments()
        AttentionBiLSTMExperiments().do_experiments()

