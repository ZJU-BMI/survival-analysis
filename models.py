import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
from sklearn.base import BaseEstimator
import time
import datetime

# 单向LSTM
class BasicLSTMModel(BaseEstimator):
    def __init__(self, time_steps, num_features, n_output, lstm_size, batch_size=64, epochs=1000, output_n_epoch=10,
                 learning_rate=0.01, max_loss=0.5, max_pace=0.01, ridge=0.0, dropout=0.8,
                 optimizer=tf.train.AdamOptimizer, name='BasicLSTMMode'):
        self._time_steps = time_steps
        self._num_features = num_features
        self._n_output = n_output
        self._lstm_size = lstm_size
        self._batch_size = batch_size
        self._epochs = epochs
        self._output_n_epoch = output_n_epoch
        self._learning_rate = learning_rate
        self._max_loss = max_loss
        self._max_pace = max_pace
        self._ridge = ridge
        self._dropout = dropout
        self._optimizer = optimizer
        self._name = name
        print("lstm_size=", lstm_size, "learning_rate=", learning_rate, "max_loss=", max_loss, "name=", name)

        with tf.variable_scope(self._name):
            self._x = tf.placeholder(tf.float32, [None, time_steps, num_features], name="input")
            self._y = tf.placeholder(tf.float32, [None, time_steps, n_output], name="label")  # 注意区别： 输出是三维tensor
            self._t = tf.placeholder(tf.float32, [None, time_steps, n_output], 'time')
            self._sess = tf.Session()
            self._hidden_layer()
            # （m,time_steps,hidden_size）->(m,time_steps,1)
            self._w_trans = tf.Variable(tf.truncated_normal([2 * self._lstm_size, self._n_output], stddev=1.0),
                                        name='output_weight')
            self._v = tf.tile(tf.reshape(self._w_trans, [-1, 2 * self._lstm_size, self._n_output]),
                              [tf.shape(self._x)[0], 1, 1])
            bias = tf.Variable(tf.random_normal([n_output]), name='output_bias')
            self._output = tf.matmul(self._hidden, self._v) + bias
            mask, _ = self._length()
            mask = tf.reshape(mask, [-1, self._time_steps, 1])
            self._pred = tf.nn.sigmoid(self._output)
            self._pred = tf.multiply(self._pred,mask)
            self._loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self._y, logits=self._pred),
                                        name='loss')

            # self._loss_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self._y, logits=self._pred))
            # loss = self.log_likelihood()
            # self._loss = tf.add(self._loss_1, loss, name='loss')
            # regularization
            if ridge != 0:
                for trainable_variables in tf.trainable_variables(self._name):
                    self._loss += tf.contrib.layers.l2_regularizer(ridge)(trainable_variables)

            self._train_op = optimizer(learning_rate).minimize(self._loss)

    def _hidden_layer(self):
        lstm = tf.contrib.rnn.BasicLSTMCell(self._lstm_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=self._dropout)
        init_state = lstm.zero_state(tf.shape(self._x)[0], tf.float32)
        mask, length = self._length()
        self._hidden, _ = tf.nn.dynamic_rnn(lstm_dropout,
                                            self._x,
                                            sequence_length=length,
                                            initial_state=init_state)

    def _length(self):
        mask = tf.sign(tf.reduce_max(tf.abs(self._x), 2))
        length = tf.reduce_sum(mask, 1)
        length = tf.cast(length, tf.int32)
        return mask, length

    def fit(self, data_set, test_set):
        self._sess.run(tf.global_variables_initializer())
        data_set.epoch_completed = 0

        for c in tf.trainable_variables(self._name):
            print(c.name)

        print("acc\tauc\tepoch\tloss\tloss_diff\tcount")
        logged = set()
        loss = 0
        count = 0
        while data_set.epoch_completed < self._epochs:
            dynamic_features, time,labels = data_set.next_batch(self._batch_size)
            self._sess.run(self._train_op, feed_dict={self._x: dynamic_features,
                                                      self._y: labels})
            if data_set.epoch_completed % self._output_n_epoch == 0 and data_set.epoch_completed not in logged:
                logged.add(data_set.epoch_completed)
                loss_prev = loss
                loss = self._sess.run(self._loss, feed_dict={self._x: data_set.dynamic_features,
                                                             self._y: data_set.labels})
                loss_diff = loss_prev - loss
                y_score = self.predict(test_set)
                y_score = y_score.reshape([-1, 1])
                test_labels = test_set.labels
                test_labels = test_labels.reshape([-1, 1])
                auc = roc_auc_score(test_labels, y_score)
                # y_score_pred = [0 for j in range(len(y_score))]
                # for i in range(len(y_score)):
                #     if y_score[i] >= 0.5:
                #         y_score_pred[i] = 1
                #     else:
                #         y_score_pred[i] = 0
                # acc = accuracy_score(test_labels, y_score_pred) # 自己设定的点 不能当作是acc
                print("{}\t{}\t{}\t{}".format(auc, data_set.epoch_completed, loss, loss_diff, count))

                # 设置训练停止条件
                if loss > self._max_loss:
                    count = 0
                else:
                    if loss_diff > self._max_pace:
                        count = 0
                    else:
                        count += 1
                if count > 9:
                    break

    def predict(self, test_set):
        # loss = self._sess.run(self._loss, feed_dict={self._x: test_set.dynamic_features[:,:,1:],
        #                                              self._y: test_set.labels})
        #                                              # self._t: test_set.dynamic_features[:,:,0].reshape(-1,1,1)})
        # print("test_loss-----" + str(loss))
        return self._sess.run(self._pred, feed_dict={self._x: test_set.dynamic_features})
                                                     # self._t:
                                                     #     test_set.dynamic_features[:,:,0].reshape(-1,self._time_steps,1)})



    @property
    def name(self):
        return self._name

    def close(self):
        self._sess.close()
        tf.reset_default_graph()


# 双向LSTM
class BidirectionalLSTMModel(BasicLSTMModel):
    def __init__(self, time_steps, num_features, n_output, lstm_size, batch_size=64, epochs=1000, output_n_epoch=10,
                 learning_rate=0.01, max_loss=0.5, max_pace=0.01, ridge=0.0, dropout=0.8,
                 optimizer=tf.train.AdamOptimizer, name="Bi-LSTM"):
        super().__init__(time_steps, num_features, n_output, lstm_size, batch_size, epochs, output_n_epoch,
                         learning_rate, max_loss, max_pace, ridge, dropout, optimizer, name)

    def _hidden_layer(self):
        self._lstm = {}
        self._lstm_dropout = {}
        self._init_state = {}
        for direction in ["forward", "backward"]:
            self._lstm[direction] = tf.contrib.rnn.BasicLSTMCell(self._lstm_size)
            self._init_state[direction] = self._lstm[direction].zero_state(tf.shape(self._x)[0], tf.float32)
        for direction in ["forward", "backward"]:
            self._lstm_dropout[direction] = tf.contrib.rnn.DropoutWrapper(self._lstm[direction],
                                                                          output_keep_prob=self._dropout)
        mask, length = self._length()
        self._hidden, _ = tf.nn.bidirectional_dynamic_rnn(self._lstm_dropout["forward"],
                                                          self._lstm_dropout["backward"],
                                                          self._x,
                                                          sequence_length=length,
                                                          initial_state_fw=self._init_state["forward"],
                                                          initial_state_bw=self._init_state["backward"])

        self._hidden = tf.concat(self._hidden, axis=2)  # n_samples×time_steps×2lstm_size→n_samples×2lstm_size


# 添加global attention机制的LSTM
class AttentionLSTMModel(BidirectionalLSTMModel):
    def __init__(self, time_steps=5, num_features=94, lstm_size=128, n_output=1, batch_size=64, epochs=1000,
                 output_n_epoch=10, learning_rate=0.01, max_loss=0.5, max_pace=0.01, ridge=0.0, dropout=0.8,
                 optimizer=tf.train.AdamOptimizer, name="AttentionLSTM"):
        self._time_steps = time_steps
        self._num_features = num_features
        self._lstm_size = lstm_size
        self._n_output = n_output
        self._batch_size = batch_size
        self._epochs = epochs
        self._output_n_epoch = output_n_epoch
        self._learning_rate = learning_rate
        self._max_loss = max_loss
        self._max_pace = max_pace
        self._ridge = ridge
        self._dropout = dropout
        self._optimizer = optimizer
        self._name = name
        print("learning_rate=", learning_rate, "max_loss=", max_loss, "max_pace=", max_pace, "name=", name)

        with tf.variable_scope(self._name):
            self._x = tf.placeholder(dtype=tf.float32, shape=[None, time_steps, num_features], name='input')
            self._y = tf.placeholder(dtype=tf.float32, shape=[None, time_steps, n_output], name='label')
            self._t = tf.placeholder(dtype=tf.float32, shape=[None, time_steps, n_output], name='time')
            self._sess = tf.Session()
            self._w = tf.Variable(tf.truncated_normal([num_features, num_features], stddev=0.1),
                                  name='attention_weight')
            self._global_attention_mechanism()
            self._hidden_layer()
            self._w_trans = tf.Variable(tf.truncated_normal([2 * self._lstm_size, self._n_output], stddev=1.0),
                                        name='output_weight')
            self._v = tf.tile(tf.reshape(self._w_trans, [-1, 2 * self._lstm_size, self._n_output]),
                              [tf.shape(self._x)[0], 1, 1])
            bias = tf.Variable(tf.random_normal([n_output]))
            self._output = tf.matmul(self._hidden, self._v) + bias
            mask, _ = self._length()
            mask = tf.reshape(mask, [-1, self._time_steps, 1])
            # 将激活函修改成tanh-->softsign(更快且不容易饱和)
            self._prediction = tf.nn.sigmoid(self._output)
            self._pred = tf.multiply(self._prediction, mask)

            # self._loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self._y,logits=self._pred),
            #                             name='loss')
            self._loss_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self._y,
                                                                                  logits=self._pred))
            neg_likelihood = self.log_likelihood()

            self._loss = tf.add(self._loss_1,neg_likelihood, name='loss')
            if ridge != 0:
                for trainable_variables in tf.trainable_variables(self._name):
                    self._loss += tf.contrib.layers.l2_regularizer(ridge)(trainable_variables)

            self._train_op = optimizer(learning_rate).minimize(self._loss)
            self._save = tf.train.Saver()

    def _global_attention_mechanism(self):
        """
            global attention : return self._z 
        """
        # attention_weight tensor
        w_x = tf.tile(tf.reshape(self._w, [-1, self._num_features, self._num_features]), [tf.shape(self._x)[0], 1, 1])
        # a11...a1m in the graph
        w_a = tf.matmul(self._x, w_x)
        # softmax in the graph
        self._w_z = tf.nn.softmax(w_a, 2)
        # get the attention output
        self._z = tf.multiply(self._x, self._w_z)
        self._z = tf.add(self._x, self._z)

    def _hidden_layer(self):
        self._lstm = {}
        self._init_state = {}
        for direction in ['forward', 'backward']:
            self._lstm[direction] = tf.contrib.rnn.BasicLSTMCell(self._lstm_size)
            self._init_state[direction] = self._lstm[direction].zero_state(tf.shape(self._x)[0], tf.float32)
        for direction in ["forward", "backward"]:
            self._lstm[direction] = tf.contrib.rnn.DropoutWrapper(self._lstm[direction],
                                                                  output_keep_prob=self._dropout)
        mask, length = self._length()
        self._hidden, _ = tf.nn.bidirectional_dynamic_rnn(self._lstm["forward"],
                                                          self._lstm["backward"],
                                                          self._z,
                                                          sequence_length=length,
                                                          initial_state_fw=self._init_state['forward'],
                                                          initial_state_bw=self._init_state['backward'])
        self._hidden = tf.concat(self._hidden, axis=2)

    def log_likelihood(self):  # 实现加入时间的loss
        risk = tf.reshape(self._pred, [-1])
        time = tf.reshape(self._t,[-1])
        E = tf.reshape(self._y, [-1])
        sort_idx = tf.argsort(time, direction='DESCENDING')
        E = tf.gather(E, sort_idx)
        risk = tf.gather(risk, sort_idx)
        hazard_ratio = tf.exp(risk)
        log_risk = tf.log(tf.cumsum(hazard_ratio))
        uncensored_likelihood = risk - log_risk
        censored_likelihood = tf.multiply(uncensored_likelihood, E)
        # num_observed_events = tf.reduce_sum(E)
        # neg_likelihood = -tf.reduce_sum(censored_likelihood) * 0.0000001
        neg_likelihood = -tf.reduce_sum(censored_likelihood) * 0.00001
        print(neg_likelihood)
        return neg_likelihood

    def fit(self, data_set, test_set):
        self._sess.run(tf.global_variables_initializer())
        data_set.epoch_completed = 0

        for c in tf.trainable_variables(self._name):
            print(c.name)

        print("acc\tauc\tepoch\tloss\tloss_diff\tcount")
        logged = set()
        loss = 0
        count = 0
        while data_set.epoch_completed < self._epochs:
            dynamic_features, time, labels = data_set.next_batch(self._batch_size)
            self._sess.run(self._train_op,
                           feed_dict={self._x: dynamic_features,
                                      self._y: labels,
                                      self._t: time.reshape(-1, dynamic_features.shape[1], 1)})
            # self._sess.run(self._train_op,
            #                feed_dict={self._x: dynamic_features[:, :, 1:],
            #                           self._y: labels})

            if data_set.epoch_completed % self._output_n_epoch == 0 and data_set.epoch_completed not in logged:
                logged.add(data_set.epoch_completed)
                loss_prev = loss
                loss = self._sess.run(self._loss,
                                      feed_dict={self._x: data_set.dynamic_features,
                                                 self._y: data_set.labels,
                                                 self._t:
                                                     data_set.time.reshape(-1,dynamic_features.shape[1],1)})

                # loss = self._sess.run(self._loss,
                #                       feed_dict={self._x: data_set.dynamic_features[:, :,1:].reshape(-1, self._time_steps,self._num_features),
                #                                  self._y: data_set.labels})
                loss_diff = loss_prev - loss
                y_score = self.predict(test_set)
                y_score = y_score.reshape([-1, ])
                test_labels = test_set.labels
                test_labels = test_labels.reshape([-1, ])
                auc = roc_auc_score(test_labels, y_score)

                print("{}\t{}\t{}\t{}\t{}\t{}".format(data_set.epoch_completed, auc, loss, loss_diff, count,datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                # 设置训练停止条件
                if loss > self._max_loss:
                    count = 0
                else:
                    if loss_diff > self._max_pace:
                        count = 0
                    else:
                        count += 1
                if count > 9:
                    break
        # save_path = self._save.save(self._sess, self._name + "model/save_net" +time.strftime("%m-%d-%H-%M-%S", time.localtime()) + ".ckpt")
        # t = time.localtime()
        t = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
        save_path = self._save.save(self._sess, self._name + "model/save_net" +t) + ".ckpt"
        print("Save to path: ", save_path)

    def predict(self, test_set):
        loss = self._sess.run(self._pred,feed_dict={self._x:test_set.dynamic_features,
                                                    self._t:test_set.time})

        # loss = self._sess.run(self._pred,feed_dict={self._x:test_set.dynamic_features[:,:,1:]})
        return loss

    def attention_analysis(self, test_dynamic, model):
        #   输入test_set, 读取模型并返回attention的weight
        saver = tf.train.Saver()
        saver.restore(self._sess, self._name + "model/" + model)
        prob = self._sess.run(self._pred, feed_dict={self._x: test_dynamic[:, :, 1:],
                                                     self._t: test_dynamic[:,:,0].reshape(-1,5,1)})
        # prob = self._sess.run(self._pred, feed_dict={self._x: test_dynamic[:, :, 1:]})
        # attention_signals = self._sess.run(self._w_z, feed_dict={self._x: test_dynamic[:, :,1:]})
        attention_signals = self._sess.run(self._w_z, feed_dict={self._x: test_dynamic[:, :, 1:93],
                                                                 self._t: test_dynamic[:,:,0].reshape(-1,5,1)})
        return prob, attention_signals.reshape([-1, self._time_steps, self._num_features])

    # add the neg-partial-likelihood loss function
    # def log_likelihood(self):
    #     risk = tf.reshape(self._pred, [-1])
    #     E = tf.reshape(self._y,[-1])
    #     sort_idx = tf.argsort(E,direction='DESCENDING')
    #     E = tf.gather(E,sort_idx)
    #     risk = tf.gather(risk,sort_idx)
    #     hazard_ratio = tf.exp(risk)
    #     log_risk = tf.log(tf.cumsum(hazard_ratio))
    #     uncensored_likelihood = risk - log_risk
    #     censored_likelihood = tf.multiply(uncensored_likelihood, E)
    #     # num_observed_events = tf.reduce_sum(E)
    #     neg_likelihood = -tf.reduce_sum(censored_likelihood) * 0.000001
    #     return neg_likelihood


class LogisticRegression(object):
    def __init__(self, time_steps, num_features, n_output, batch_size=64, epochs=1000, output_n_epoch=10,
                 learning_rate=0.01, max_loss=0.5, max_pace=0.01, ridge=0.0, dropout=0.8,
                 optimizer=tf.train.AdamOptimizer, name="LogisticRegression"):
        self._time_steps = time_steps
        self._num_features = num_features
        self._n_output = n_output
        self._batch_size = batch_size
        self._epochs = epochs
        self._output_n_epoch = output_n_epoch
        self._learning_rate = learning_rate
        self._max_loss = max_loss
        self._max_pace = max_pace
        self._ridge = ridge
        self._dropout = dropout
        self._optimizer = optimizer
        self._name = name
        print("learning_rate=", learning_rate, "max_loss=", max_loss,
              "max_loss=", max_loss, "max_pace=", max_pace, "name=", name)
        with tf.variable_scope(self._name):
            self._x = tf.placeholder(tf.float32, [None, num_features], name="input")
            self._y = tf.placeholder(tf.float32, [None, 1], name="label")
            self._sess = tf.Session()
            self._hidden_layer()
            self._output = tf.contrib.layers.fully_connected(self._hidden, n_output,
                                                             activation_fn=tf.identity)
            self._pred = tf.nn.sigmoid(self._output, name="pred")
            self._loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self._y, logits=self._pred),
                                        name="loss")
            self._train_op = optimizer(learning_rate).minimize(self._loss)

    def _hidden_layer(self):
        self._hidden = self._x

    def fit(self, data_set, test_set):
        self._sess.run(tf.global_variables_initializer())
        data_set.epoch_completed = 0
        for c in tf.trainable_variables(self._name):
            print(c.name)

        print("acc\tauc\tepoch\tloss\tloss_diff\tcount")
        logged = set()
        loss = 0
        count = 0
        while data_set.epoch_completed < self._epochs:
            dynamic_features, labels = data_set.next_batch(self._batch_size)
            self._sess.run(self._train_op, feed_dict={self._x: dynamic_features,
                                                      self._y: labels})
            if data_set.epoch_completed % self._output_n_epoch == 0 and data_set.epoch_completed not in logged:
                logged.add(data_set.epoch_completed)
                loss_prev = loss
                loss = self._sess.run(self._loss, feed_dict={self._x: data_set.dynamic_features,
                                                             self._y: data_set.labels})
                loss_diff = loss_prev - loss
                y_score = self.predict(test_set)  # 此处计算和打印auc仅供调参时观察auc变化用，可删除，与最终输出并无关系
                auc = roc_auc_score(test_set.labels, y_score)
                y_score_pred = [0 for j in range(len(y_score))]
                for i in range(len(y_score)):
                    if y_score[i] >= 0.5:
                        y_score_pred[i] = 1
                    else:
                        y_score_pred[i] = 0
                acc = accuracy_score(test_set.labels, y_score_pred)
                print("{}\t{}\t{}\t{}\t{}\t{}".format(acc, auc, data_set.epoch_completed, loss, loss_diff, count),
                      time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

                # 训练停止条件
                if loss > self._max_loss:
                    count = 0
                else:
                    if loss_diff > self._max_pace:
                        count = 0
                    else:
                        count += 1
                if count > 9:
                    break

    def predict(self, test_set):
        return self._sess.run(self._pred, feed_dict={self._x: test_set.dynamic_features,
                                                     self._y: test_set.labels})

    @property
    def name(self):
        return self._name

    def close(self):
        self._sess.close()
        tf.reset_default_graph()


class SelfAttentionLSTMModel(BidirectionalLSTMModel):
    def __init__(self, time_steps, num_features, lstm_size, n_output, batch_size=64, epochs=1000, output_n_epoch=10,
                 learning_rate=0.01, max_loss=0.5, max_pace=0.01, ridge=0.0, dropout=0.8,
                 optimizer=tf.train.AdamOptimizer, name="LocalAttentionLSTM"):
        self._time_steps = time_steps
        self._num_features = num_features
        self._lstm_size = lstm_size
        self._n_output = n_output
        self._batch_size = batch_size
        self._epochs = epochs
        self._output_n_epoch = output_n_epoch
        self._learning_rate = learning_rate
        self._max_loss = max_loss
        self._max_pace = max_pace
        self._ridge = ridge
        self._dropout = dropout
        self._optimizer = optimizer
        self._name = name
        print("learning_rate=", learning_rate, "max_loss=", max_loss, "max_pace=", max_pace, "name=", name)

        with tf.variable_scope(self._name):
            self._x = tf.placeholder(tf.float32, [None, time_steps, num_features], 'input')
            self._y = tf.placeholder(tf.float32, [None, time_steps, n_output], 'label')
            self._sess = tf.Session()
            self._self_attention_mechanism()
            self._hidden_layer()
            self._w_trans = tf.Variable(tf.truncated_normal([2 * self._lstm_size, self._n_output], stddev=0.1),
                                        name='output_weight')
            self._v = tf.tile(tf.reshape(self._w_trans, [-1, 2 * self._lstm_size, self._n_output]),
                              [tf.shape(self._x)[0], 1, 1])
            bias = tf.Variable(tf.random_normal([n_output]))
            self._output = tf.matmul(self._hidden, self._v) + bias
            self._pred = tf.nn.sigmoid(self._output)
            self._loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self._y, logits=self._pred),
                                        name="loss")

            if ridge != 0:
                for trainable_variables in tf.trainable_variables(self._name):
                    self._loss += tf.contrib.layers.l2_regularizer(ridge)(trainable_variables)

            self._train_op = optimizer(learning_rate).minimize(self._loss)
            self._save = tf.train.Saver()

    def _self_attention_mechanism(self):
        """
            self attention : return self._z
        """
        dims = 64
        self._q = tf.Variable(tf.truncated_normal([self._num_features, dims], stddev=0.1), name='self_attention_w')
        self._k = tf.Variable(tf.truncated_normal([self._num_features, dims], stddev=0.1), name='self_attention_k')
        self._v = tf.Variable(tf.truncated_normal([self._num_features, dims], stddev=0.1), name='self_attention_v')
        self._w0 = tf.Variable(tf.truncated_normal([dims, self._num_features], stddev=0.1), name='self_attention_w0')
        q_trans = tf.tile(tf.reshape(self._q, [-1, self._num_features, dims]), [tf.shape(self._x)[0], 1, 1])
        k_trans = tf.tile(tf.reshape(self._k, [-1, self._num_features, dims]), [tf.shape(self._x)[0], 1, 1])
        v_trans = tf.tile(tf.reshape(self._v, [-1, self._num_features, dims]), [tf.shape(self._x)[0], 1, 1])
        # q_trans = tf.Variable(tf.truncated_normal([tf.shape(self._x)[0],self._num_features,dims]))
        # k_trans = tf.Variable(tf.truncated_normal([tf.shape(self._x)[0],self._num_features,dims]))
        # v_trans = tf.Variable(tf.truncated_normal([tf.shape(self._x)[0],self._num_features,dims]))
        w0 = tf.tile(tf.reshape(self._w0, [-1, dims, self._num_features]), [tf.shape(self._x)[0], 1, 1])
        q = tf.matmul(self._x, q_trans)
        k = tf.matmul(self._x, k_trans)
        v = tf.matmul(self._x, v_trans)
        self._m = tf.nn.softmax(tf.matmul(tf.matmul(q, tf.transpose(k, [0, 2, 1]))/8, v), 2)
        self._z = tf.matmul(self._m, w0)
        self._z = tf.add(self._z,self._x)

    def _hidden_layer(self):
        self._lstm = {}
        self._init_state = {}
        for direction in ['forward', 'backward']:
            self._lstm[direction] = tf.contrib.rnn.BasicLSTMCell(self._lstm_size)
            self._init_state[direction] = self._lstm[direction].zero_state(tf.shape(self._x)[0], tf.float32)
        for direction in ["forward", "backward"]:
            self._lstm[direction] = tf.contrib.rnn.DropoutWrapper(self._lstm[direction],
                                                                  input_keep_prob=self._dropout,
                                                                  output_keep_prob=self._dropout)
        mask, length = self._length()
        self._hidden, _ = tf.nn.bidirectional_dynamic_rnn(self._lstm['forward'],
                                                          self._lstm['backward'],
                                                          self._z,
                                                          sequence_length=length,
                                                          initial_state_fw=self._init_state['forward'],
                                                          initial_state_bw=self._init_state['backward'])
        self._hidden = tf.concat(self._hidden, axis=2)

    def fit(self, data_set, test_set):
        self._sess.run(tf.global_variables_initializer())
        data_set.epoch_completed = 0

        for c in tf.trainable_variables(self._name):
            print(c.name)

        print("acc\tauc\tepoch\tloss\tloss_diff\tcount")
        logged = set()
        loss = 0
        count = 0
        while data_set.epoch_completed < self._epochs:
            dynamic_features, time,labels = data_set.next_batch(self._batch_size)
            self._sess.run(self._train_op, feed_dict={self._x: dynamic_features,
                                                      self._y: labels})
            if data_set.epoch_completed % self._output_n_epoch == 0 and data_set.epoch_completed not in logged:
                logged.add(data_set.epoch_completed)
                loss_prev = loss
                loss = self._sess.run(self._loss, feed_dict={self._x: data_set.dynamic_features,
                                                             self._y: data_set.labels})
                loss_diff = loss_prev - loss
                y_score = self.predict(test_set)
                y_score = y_score.reshape([-1, 1])
                test_labels = test_set.labels
                test_labels = test_labels.reshape([-1, 1])
                auc = roc_auc_score(test_labels, y_score)
                y_score_pred = [0 for j in range(len(y_score))]
                for i in range(len(y_score)):
                    if y_score[i] >= 0.5:
                        y_score_pred[i] = 1
                    else:
                        y_score_pred[i] = 0
                acc = accuracy_score(test_labels, y_score_pred)
                print("{}\t{}\t{}\t{}\t{}\t{}".format(acc, auc, data_set.epoch_completed, loss, loss_diff, count))

                # 设置训练停止条件
                if loss > self._max_loss:
                    count = 0
                else:
                    if loss_diff > self._max_pace:
                        count = 0
                    else:
                        count += 1
                if count > 9:
                    break
        # save_path = self._save.save(self._sess, self._name + "model/save_net" +
        #                             time.strftime("%m-%d-%H-%M-%S", time.localtime())
        #                             + ".ckpt")
        t = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
        save_path = self._save.save(self._sess, self._name + "model/save_net" + t) + ".ckpt"
        print("Save to path: ", save_path)
