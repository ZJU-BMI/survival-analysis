import tensorflow as tf
import numpy as np
from data import read_data,read_WHAS_dataset
from tensorflow.contrib.rnn import GRUCell,BasicLSTMCell
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from sklearn.metrics import roc_auc_score,accuracy_score
import time


class GnnLSTMSurV(object):
    def __init__(self,batch_size,time_steps,num_features,n_output,
                 hidden_size,epoch,learning_rate,max_loss, max_pace, ridge,dropout,feature_dims,name="GNN_LSTM_Surv"):
        self._batch_size = batch_size
        self._time_steps = time_steps
        self._num_features = num_features
        self._n_output = n_output
        self._hidden_size = hidden_size
        self._epoch = epoch
        self._learning_rate = learning_rate
        self._max_loss = max_loss
        self._max_pace = max_pace
        self._ridge = ridge
        self._dropout = dropout
        self._name = name
        self._feature_dims = feature_dims
        print("learning_rate=",learning_rate,"lstm_size=",hidden_size,"epoch=",epoch,"batch_size=",batch_size,"ridge=",ridge)

        with tf.variable_scope(self._name):
            optimizer = tf.train.GradientDescentOptimizer
            self._sess = tf.Session()
            with tf.name_scope("input"):
                self._x = tf.placeholder(tf.float32,shape=[None,time_steps,num_features,feature_dims],name="input_x")
                self._y = tf.placeholder(tf.float32,shape=[None,time_steps,n_output],name="input_y")
                self._t = tf.placeholder(tf.float32,shape=[None,time_steps,n_output],name="input_t")
            attention_output = self.global_attention_mechanism(self._x)
            INPUT = tf.add(self._x,attention_output)
            all_states = self.state_update(INPUT, self._hidden_size)
            states = []
            for i in range(self._time_steps):
                states.append(tf.concat((all_states[0][i],all_states[1][i]),axis=1))
            states = tf.transpose(states,[1,0,2])
            self._output = self.output(states)
            mask,length = self.length()
            mask = tf.reshape(mask,[-1,time_steps,1])
            prediction = tf.nn.sigmoid(self._output)
            self._prediction = tf.multiply(prediction,mask)
            loss_neg_likelihood = self.partial_log_likelihood()
            # self._loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self._y,logits=self._prediction),name='loss')
            self._loss_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self._y,logits=self._prediction))
            self._loss = tf.add(loss_neg_likelihood,self._loss_1,name="loss")

            if ridge != 0:
                for trainable_variable in tf.trainable_variables(self._name):
                    self._loss += tf.contrib.layers.l2_regularizer(ridge)(trainable_variable)
            self._train_op = optimizer(self._learning_rate).minimize(self._loss)
            self._save = tf.train.Saver()
            writer = tf.summary.FileWriter("logs/",self._sess.graph)

    with tf.name_scope("attention_mechanism"):
        def global_attention_mechanism(self,x):
            time_steps = self._time_steps
            num_features = self._num_features
            feature_dims = self._feature_dims
            self._w = tf.Variable(tf.truncated_normal([num_features,feature_dims]))
            self._w_ = tf.nn.softmax(self._w,axis=1)
            self._w_z = tf.tile(tf.reshape(self._w_,[-1,1,num_features,feature_dims]),[tf.shape(x)[0],self._time_steps,1,1])
            self._z = tf.multiply(x,self._w_z)
            # self._w = tf.Variable(tf.truncated_normal([feature_dims*num_features, feature_dims*num_features],stddev=0.1),name="attention_weight")
            # w_x = tf.tile(tf.reshape(self._w,[-1,feature_dims*num_features,num_features*feature_dims]),[tf.shape(x)[0],1,1])
            # w_a = tf.matmul(tf.reshape(self._x,[-1,self._time_steps,feature_dims*num_features]),w_x)
            # w_a = tf.reshape(w_a,[-1,self._time_steps,num_features,feature_dims])
            # self._w_z = tf.nn.softmax(w_a,axis=2)
            # self._z = tf.multiply(x,self._w_z)
            return self._z

    def fit(self,data_set,test_set):
        self._sess.run(tf.global_variables_initializer())
        data_set.epoch_completed = 0

        for c in tf.trainable_variables(self._name):
            print(c)
        print("epoch\tauc\tloss\tloss_diff\tcount")
        logged = set()
        loss = 0
        count = 0
        while data_set.epoch_completed < self._epoch:
            train_features,train_time,train_y = data_set.next_batch(self._batch_size)
            self._sess.run(self._train_op,feed_dict={self._x:train_features,
                                                     self._t:train_time,
                                                     self._y:train_y})
            if data_set.epoch_completed % self._n_output==0 and data_set.epoch_completed not in logged:
                logged.add(data_set.epoch_completed)
                loss_pre = loss
                loss = self._sess.run(self._loss,feed_dict={self._x:train_features,
                                                            self._t:train_time,
                                                            self._y:train_y})
                loss_diff = loss_pre - loss
                y_score,test_label= self.predict(test_set)
                y_score = np.array(y_score)
                test_label = np.array(test_label)
                test_y_score = np.zeros(shape=(0))
                test_label_all = np.zeros(shape=(0))
                for i in range(y_score.shape[0]):
                    test_y_score = np.concatenate((test_y_score,y_score[i].reshape(-1,)))
                    test_label_all = np.concatenate((test_label_all,test_label[i].reshape(-1,)))
                test_y = test_label_all.reshape([-1,])
                auc = roc_auc_score(test_y,test_y_score)
                print("{}\t{}\t{}\t{}\t{}".format(data_set.epoch_completed,auc,loss,loss_diff,count),
                      time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))

                # 设置训练停止 条件
                if loss > self._max_loss:
                    count = 0
                else:
                    if loss_diff> self._max_pace:
                        count = 0
                    else:
                        count += 1
                if count>9:
                    break
        save_path = self._save.save(self._sess,self._name+"model/save_net" +
                                    time.strftime("%m-%d-%H-%M-%S",time.localtime())+".ckpt")
        print("save to path: ",save_path)

    # Todo:修改函数，分批读取
    def predict(self,test_set):
        # logged = set()
        # prediction = []
        # while test_set.epoch_completed < 1:
        #     test_x, test_t,test_y = test_set.next_batch(self._batch_size)
        #     pre_batch = self._sess.run(self._prediction,feed_dict={self._x:test_x,self._t:test_t})
        #     prediction.append(pre_batch)
        #     if test_set.epoch_completed % self._output == 0 and test_set.epoch_completed not in logged:
        #         logged.add(test_set.epoch_completed)
        #         pre = self._sess.run(self._prediction,feed_dict={self._x:test_x,self._t:test_t})
        #         prediction.append(pre)
        # return prediction
        all_prediction = list()
        all_labels = list()
        epochs = int(test_set.dynamic_features.shape[0]/self._batch_size)
        if epochs*self._batch_size < test_set.dynamic_features.shape[0]:
            epochs += 1
        for i in range(epochs):
            test_x,test_t,test_y = test_set.predict_next_batch(self._batch_size)
            # test_x,test_t,test_y = test_set.next_batch(self._batch_size)
            prediction = self._sess.run(self._prediction,feed_dict={self._x:test_x,self._t:test_t})
            all_prediction.append(prediction)
            all_labels.append(test_y)
        return all_prediction,all_labels

    def partial_log_likelihood(self):
        risk = tf.reshape(self._prediction,[-1])
        time = tf.reshape(self._t,[-1])
        E = tf.reshape(self._y,[-1])
        sort_idx = tf.argsort(time,direction='DESCENDING')
        E = tf.gather(E,sort_idx)
        hazard_ratio = tf.exp(risk)
        log_risk = tf.log(tf.cumsum(hazard_ratio))
        uncensored_likelihood = risk-log_risk
        censored_likelihood = tf.multiply(uncensored_likelihood,E)
        neg_likelihood = -tf.reduce_sum(censored_likelihood) * 0.00001
        return neg_likelihood

# get tbe attention weight before embedding
    with tf.name_scope("attention_analysis"):
        def attention_analysis(self,test_x,test_t,model):
            # init = tf.global_variables_initializer()
            # self._sess.run(init)
            # self._sess.run(tf.initialize_all_variables())
            # read the attention between the nodes and the features attention
            saver = self._save
            saver.restore(self._sess,self._name+"model/"+model)
            print(saver._var_list)
            prediction = self._sess.run(self._prediction,feed_dict={self._x:test_x,self._t:test_t})
            # attention_signals = self._sess.run(self._w_z,feed_dict={self._x:test_x,self._t:test_t})
            attention_weights = self._sess.run(self._w_,feed_dict={self._x:test_x,self._t:test_t})
            graph = self._sess.run(self._graph_adjacency,feed_dict={self._x:test_x,self._t:test_t})
            return prediction,attention_weights.reshape(-1,self._num_features,self._feature_dims),graph


    with tf.name_scope("init_graph"):
        def _init_graph(self,x):
            """
            # construct graph 也就是邻接矩阵的构建
            :param x: [batch_size,time_steps,num_features,feature_dims]
            :return graph: [batch__size,time_steps,num_features,num_features] denotes the adjacency matrix
            """
            time_steps = self._time_steps
            num_node = self._num_features
            node_dims = self._feature_dims
            x = tf.reshape(x, [-1, tf.shape(x)[2], tf.shape(x)[3]])
            num_all = tf.shape(x)[0]  # 总batch数目
            a = tf.tile(x, [1, num_node, 1])
            b = tf.tile(x, [1, 1, num_node])
            a = tf.reshape(a, [num_all, num_node, num_node, node_dims])
            b = tf.reshape(b, [num_all, num_node, num_node, node_dims])
            m = tf.concat((a, b), axis=3)
            m = tf.reshape(m, [num_all, num_node * num_node, 2 * node_dims])
            with tf.variable_scope("w"):
                w_w = tf.get_variable(name="weight_w",shape=[2 * node_dims, 1],initializer=tf.truncated_normal_initializer(stddev=0.1))
            with tf.variable_scope("b"):
                w_b = tf.get_variable(name="weight_b",shape=[1],initializer=tf.random_normal_initializer([1.0]))
            w_w_all = tf.tile(w_w, [num_all, 1])
            w_w_all = tf.reshape(w_w_all, [num_all, 2 * node_dims, 1])
            w_b_all = tf.tile(w_b, [num_all * num_node])
            w_b_all = tf.reshape(w_b_all, [num_all, num_node, 1])
            w_b_all = tf.cast(w_b_all, tf.float32)
            weight = tf.reshape(tf.matmul(m, w_w_all), [num_all, num_node, num_node]) + w_b_all
            weight = tf.linalg.band_part(weight,0,-1)
            # weight = tf.nn.softmax(weight,axis=2)
            weight_trans = tf.transpose(weight,perm=[0,2,1])
            weight_ = tf.add(weight,weight_trans)
            adjacency = tf.ones([num_node, num_node]) - tf.eye(num_node, num_columns=num_node)
            adjacency_all = tf.reshape(tf.tile(adjacency, [num_all, 1]), [num_all, num_node, num_node])
            graph_ = tf.multiply(weight_, adjacency_all)
            # graph = tf.reshape(graph,[-1,time_steps,num_node,num_node])
            graph = tf.reshape(tf.nn.softmax(graph_,axis=1),[-1,time_steps,num_node,num_node])
            self._graph_adjacency = graph_
            return graph
    with tf.name_scope("state_aggregation"):
        def gnn_message_pass(self,x,graph):
            """
            :param x: [batch_size,num_node,node_dims]
            :param graph: [batch_size,num_node,num_node]
            :return state_aggregation；[batch_size,num_node,hidden_size]
            """
            batch = tf.shape(x)[0]
            num_node = self._num_features
            node_dims = self._feature_dims
            graph = tf.reshape(graph,[-1,num_node,num_node])
            x = tf.reshape(x,[-1,num_node,node_dims])
            with tf.variable_scope("w"):
                w_out = tf.get_variable(name="weight_out",shape=[node_dims,node_dims],initializer=tf.truncated_normal_initializer(stddev=0.1))
            with tf.variable_scope("b"):
                w_out_b = tf.get_variable(name="weight_out_b",shape=[num_node,1],initializer=tf.random_normal_initializer())
            w_out_all = tf.reshape(tf.tile(w_out,[batch,1]),[batch,node_dims,node_dims])
            w_out_b_all = tf.reshape(tf.tile(w_out_b,[batch,node_dims]),[batch,num_node,node_dims])
            x_ = tf.add(tf.matmul(x,w_out_all),w_out_b_all)  # size=[batch_size,num_node,node_dims]
            states = tf.matmul(graph,x_)  # size=[batch_size,num_node,node_dims]  这里是所有的state*out
            with tf.variable_scope("w"):
                w_in = tf.get_variable(name="weight_in",shape=[node_dims,node_dims],initializer=tf.truncated_normal_initializer(stddev=0.1))
            with tf.variable_scope("b"):
                w_in_b = tf.get_variable(name="weight_in_b",shape=[num_node,1],initializer=tf.random_normal_initializer())
            w_in_all = tf.reshape(tf.tile(w_in,[batch,1]),[batch,node_dims,node_dims])
            # w_in_b = tf.cast(w_in_b,tf.float32)
            w_in_b_all = tf.reshape(tf.tile(w_in_b,[batch,node_dims]),[batch,num_node,node_dims])
            state_aggregation = tf.add(tf.matmul(states,w_in_all),w_in_b_all)  # 这里表示所有的state aggregation
            return state_aggregation

    def state_update(self,x,hidden_size):
        """
        :param x: embedding_layer[batch_size,time_steps,node_num,node_dims]
        :param hidden_size: lstm hidden_sie
        :return all_states:[batch_size,time_steps,node_num,hidden_size]
        """
        batch = tf.shape(x)[0]
        time_steps = self._time_steps
        num_node = self._num_features
        node_dims = self._feature_dims
        lstm_cell = BasicLSTMCell(hidden_size*num_node)
        graph = self._init_graph(x)
        all_states_fw = [0,0,0,0,0]
        all_states_bw = [0,0,0,0,0]
        # basic_lstm_cell = BasicLSTMCell(hidden_size)
        with tf.name_scope("lstm_forward"):
            for i in range(self._time_steps):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                # 取第二维度的x 和第二维度的state aggregation
                graph_ = graph[:,i,:,:]
                x_i = x[:,i,:,:]
                state_aggregation = self.gnn_message_pass(x_i,graph_)
                input = tf.reshape(x_i,[-1,num_node*node_dims])
                c = tf.reshape(state_aggregation,[-1,hidden_size*num_node])
                if i==0:
                    h = tf.zeros(shape=(tf.shape(c)))
                    # m = tf.zeros(shape=(tf.shape(c)))   # 尝试只使用LSTM
                else:
                    h = new_h
                    c = new_c
                    # m = new_m
                states = (c,h)
                # states = (m,h)
                new_h,(new_c,_) = lstm_cell(input,states)
                # state = tf.reshape(state_aggregation,[-1,hidden_size])  # 之前的state
                # input = tf.reshape(tf.gather(x,[i],axis=1),[-1,num_node,node_dims])
                # state = tf.reshape(tf.gather(state_aggregation,[i],axis=1),[-1,num_node,hidden_size])
                # _, x_ = gru_cell(input, state)  # 之前的gru_cell
                # x_ = tf.reshape(x_,[-1,batch,tf.shape(x_)[1],tf.shape(x_)[2]])
                # x = x_
                # x_ = tf.reshape(x_,[-1,batch,num_node,node_dims])
                # all_states[i] = x_
                new_h_ = tf.reshape(new_h,[-1,num_node*hidden_size])
                all_states_fw[i] = new_h_
        with tf.name_scope("lstm_backward"):
            for j in range(self._time_steps):
                tf.get_variable_scope().reuse_variables()
                l = self._time_steps -j-1
                graph_ = graph[:,l,:,:]
                x_j = x[:,l,:,:]
                state_aggregation = self.gnn_message_pass(x_j,graph_)
                input = tf.reshape(x_j,[-1, num_node*node_dims])
                c = tf.reshape(state_aggregation, [-1,hidden_size*num_node])
                if j == 0:
                    h = tf.zeros(shape=tf.shape(c))
                    # m = tf.zeros(shape=tf.shape(c))
                else:
                    h = new_h
                    c = new_c
                    # m = new_m
                states = (c,h)
                # states = (m,h)
                new_h, (new_c,_) = lstm_cell(input,states)
                new_h_ = tf.reshape(new_h,[-1, num_node*hidden_size])
                all_states_bw[l] = new_h_
        all_states = [all_states_fw, all_states_bw]
        return all_states

    with tf.name_scope("output"):
        def output(self,all_states):
            node_nums = self._num_features
            node_dims = self._feature_dims
            hidden_size = self._hidden_size
            # all_states = tf.reshape(all_states,[-1,self._time_steps,node_nums*hidden_size*2])
            all_states = tf.reshape(all_states,[-1,node_nums*hidden_size*2])
            output_weight = tf.Variable(tf.truncated_normal([node_nums*hidden_size*2,self._n_output],stddev=0.1),name="output_weight")
            output_b = tf.Variable(tf.random_normal([self._n_output]),name="output_b")
            output = tf.matmul(all_states,output_weight) + output_b
            output = tf.reshape(output,[-1, self._time_steps,1])
            return output
    with tf.name_scope("mask"):
        def length(self):
            m = tf.reduce_max(tf.abs(self._x),3)
            mask = tf.sign(tf.reduce_max(tf.abs(m),2))
            length = tf.reduce_sum(mask,1)
            length = tf.cast(length,tf.int32)
            return mask,length

    def close(self):
        self._sess.close()
        tf.reset_default_graph()













