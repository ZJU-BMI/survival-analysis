import os
import jenkspy
import matplotlib.pyplot as plt
import sklearn
import time
import xlwt
from sklearn.cluster import KMeans
import xlsxwriter
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score,f1_score,precision_score,recall_score,roc_curve
from data import read_data,DataSet
from models import BidirectionalLSTMModel,AttentionLSTMModel,LogisticRegression,SelfAttentionLSTMModel


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
bi_lstm_setup = ExperimentSetup(0.03,0.5,0.01,0.001)
ca_rnn_seup = ExperimentSetup(0.0001,0.08,0.001,0.001)
self_rnn_setup = ExperimentSetup(0.01,0.08,0.001,0.001)


def split_data_set(dynamic_features, labels):
    time_steps = dynamic_features.shape[1]
    num_features = dynamic_features.shape[2]
    train_dynamic_features = {}
    train_labels = {}
    test_dynamic_features = {}
    test_labels = {}
    num = int(dynamic_features.shape[0] / 5)
    for i in range(5):
        test_dynamic_features[i] = dynamic_features[i * num:(i + 1) * num, :, :].reshape(-1, time_steps, num_features)
        test_labels[i] = labels[i * num:(i + 1) * num, :, :].reshape(-1, time_steps, 1)
    train_dynamic_features[0] =dynamic_features[num:5*num,:,:]
    train_labels[0] = labels[num:5*num,:,:]

    train_dynamic_features[1] = np.vstack((dynamic_features[0:num, :, :],dynamic_features[2*num:5*num,:,:]))
    train_labels[1] = np.vstack((labels[0:num,:,:], labels[2*num:5*num,:,:]))

    train_dynamic_features[2] = np.vstack((dynamic_features[0:2*num, :, :],dynamic_features[3*num:5*num,:,:]))
    train_labels[2] = np.vstack((labels[0:2*num,:,:], labels[3*num:5*num,:,:]))

    train_dynamic_features[3] = np.vstack((dynamic_features[0:3*num, :, :],dynamic_features[4*num:5*num,:,:]))
    train_labels[3] = np.vstack((labels[0:3*num,:,:], labels[4*num:5*num,:,:]))

    train_dynamic_features[4] = dynamic_features[0:4*num, :, :]
    train_labels[4] = labels[0:4*num, :, :]

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
        if not os.path.exists("result_9_10_0（"):
            os.makedirs("result_9_10_0")
        self._filename = "result_9_10_0" + "/" + self._model.name + " " + time.strftime( "%Y-%m-%d-%H-%M-%S", time.localtime())

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
        if not os.path.exists("result_9_10_0"):
            os.makedirs("result_9_10_0")
        self._filename = "result_9_10_0" + "/" + self._model.name + " " + time.strftime( "%Y-%m-%d-%H-%M-%S", time.localtime())

    def do_experiments(self):
        n_output=1
        dynamic_features = self._data_set.dynamic_features
        labels = self._data_set.labels
        tol_test_index = np.zeros(shape=0, dtype=np.int32)
        tol_pred = np.zeros(shape=(0, dynamic_features.shape[1],n_output))
        tol_label = np.zeros(shape=(0, dynamic_features.shape[1],n_output), dtype=np.int32)
        train_dynamic_features, test_dynamic_features, train_labels, test_labels = split_data_set(dynamic_features, labels)
        for i in range(5):
            train_dynamic_res, train_labels_res = imbalance_preprocess(train_dynamic_features[i],train_labels[i],'lstm')
            train_set = DataSet(train_dynamic_res, train_labels_res)
            test_set = DataSet(test_dynamic_features[i],test_labels[i])
            self._model.fit(train_set, test_set)
            y_score = self._model.predict(test_set)
            tol_pred = np.vstack((tol_pred, y_score))
            tol_label = np.vstack((tol_label, test_labels[i]))
            print("Cross validation: {} of {}".format(i, 5),
                  time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        tol_test_index = np.arange(labels.shape[0]*labels.shape[1])
        evaluate(tol_test_index, tol_label, tol_pred, self._filename)
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

    def attention_analysis(self):
        attention_signals_tol = np.zeros(shape=(0,self._time_steps,self._num_features))
        models = ["save_net09-09-18-59.ckpt", "save_net09-09-19-00.ckpt",
                  "save_net09-09-19-01.ckpt", "save_net09-09-19-02.ckpt",
                  "save_net09-09-19-03.ckpt"]
        n_output = 1
        dynamic_features = self._data_set.dynamic_features
        labels = self._data_set.labels
        train_dynamic_features, test_dynamic_features, train_labels, test_labels = split_data_set(dynamic_features,
                                                                                                  labels)
        for i in range(5):
            test_set = DataSet(test_dynamic_features[i], test_labels[i])
            prob, attention_weight = self._model.attention_analysis(test_set.dynamic_features, models[i])
            attention_signals_tol = np.concatenate((attention_signals_tol, attention_weight))
        np.save("allAttentionWeight.npy",attention_signals_tol)



    def cluster_by_attention_weight(self):
        attentionWeight = np.load("allAttentionWeight.npy")
        attentionWeightArray = attentionWeight.reshape([-1,self._num_features])
        all_feature_breaks = []
        for nums in range(self._num_features):
            one_feature_breaks = jenkspy.jenks_breaks(attentionWeightArray[:,nums],nb_class=5)
            print(one_feature_breaks)
            all_feature_breaks.append(one_feature_breaks)
        np.save("all_features_breaks.npy",all_feature_breaks)

    def get_stages(self):
        all_features_breaks = np.load("all_features_breaks.npy")
        attention_weight = np.load("allAttentionWeight.npy")
        one_patient_stage = []
        one_patient_score = []
        all_patient_stage = np.zeros(shape=(0,5),dtype=np.int32)
        all_patient_score = np.zeros(shape=(0,5),dtype=np.float32)
        for patient in range(attention_weight.shape[0]):
            for visit in range(attention_weight.shape[1]):
                one_patient_features = attention_weight[patient,visit,:]
                patient_features_in_stage1 = []
                patient_features_in_stage2 = []
                patient_features_in_stage3 = []
                patient_features_in_stage4 = []
                patient_features_in_stage5 = []
                for i in range(one_patient_features.shape[0]):
                    if one_patient_features[i] < all_features_breaks[i,1]:
                        patient_features_in_stage1.append(one_patient_features[i])

                    if all_features_breaks[i,1] <= one_patient_features[i] < all_features_breaks[i,2]:
                        patient_features_in_stage2.append(one_patient_features[i])

                    if all_features_breaks[i,2] <= one_patient_features[i] < all_features_breaks[i,3]:
                        patient_features_in_stage3.append(one_patient_features[i])

                    if all_features_breaks[i,3] <= one_patient_features[i] < all_features_breaks[i,4]:
                        patient_features_in_stage4.append(one_patient_features[i])

                    if all_features_breaks[i,4] <= one_patient_features[i] <= all_features_breaks[i,5]:
                        patient_features_in_stage5.append(one_patient_features[i])

                stage1_score = np.sum(patient_features_in_stage1)
                stage2_score = np.sum(patient_features_in_stage2)
                stage3_score = np.sum(patient_features_in_stage3)
                stage4_score = np.sum(patient_features_in_stage4)
                stage5_score = np.sum(patient_features_in_stage5)
                score = [stage1_score,stage2_score,stage3_score,stage4_score,stage5_score]
                max_score = max(score)
                max_score_index = np.argmax(score)
                one_patient_score.append(max_score)
                one_patient_stage.append(max_score_index)
            all_patient_stage = np.concatenate((all_patient_stage, np.array(one_patient_stage).reshape(-1,5)))
            all_patient_score = np.concatenate((all_patient_score, np.array(one_patient_score).reshape(-1,5)))
            one_patient_stage = []
            one_patient_score = []
        np.save('all_patient_stage.npy',all_patient_stage)
        np.save('all_patient_score.npy',all_patient_score)


class SelfAttentionBiLSTMExperiments(BidirectionalLSTMExperiments):
    def __init__(self):
        super().__init__()

    def _model_format(self):
        learning_rate, max_loss, max_pace, ridge = self_rnn_setup.all
        self._model = SelfAttentionLSTMModel(num_features=self._num_features,
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
        # AttentionBiLSTMExperiments().do_experiments()
        # SelfAttentionBiLSTMExperiments().do_experiments()
        AttentionBiLSTMExperiments().get_stages()