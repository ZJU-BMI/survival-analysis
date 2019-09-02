import numpy as np
import tensorflow as tf
# x = [[[2.0,3.0],[3.0,4.0]],
#      [[1.5,2.0],[5.0,8.5]],
#      [[0.5,2.0],[3.5,5.2]]]
# x = np.array(x)
# print(x.shape)
# s = np.zeros(shape=(x.shape[0],2,1))
# for i in range(x.shape[0]):
#     m = x[i,:,:]
#     print(m)
#     y = tf.contrib.layers.fully_connected(m,1,activation_fn=tf.identity)
#     pred = tf.nn.sigmoid(y, name="pred")
#     pred = tf.reshape(pred,[1,2,1])
#
#     # s[i,:,:] = pred
#     indices = tf.constant([[i]])
#     shape = tf.constant([i+1,2,1])
#     scatter = tf.scatter_nd(indices, pred, shape)
#     sess = tf.Session()
#     sess.run(tf.global_variables_initializer())
#     scatter = scatter.eval(session=sess)
#     print(scatter)
a = tf.constant(np.arange(1, 13, dtype=np.float32), shape=[2, 2, 3])
b = tf.constant(np.arange(1, 13, dtype=np.float32), shape=[2, 3, 2])
sess = tf.Session()
c = tf.matmul(a,b)
print(c.eval(session=sess))






