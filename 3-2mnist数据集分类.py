import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
#载入数据集
mnist=input_data.read_data_sets('mnist_data',one_hot=True)
#每个批次大小
batch_size=80
#计算一共有多少个批次
n_batch=mnist.train.num_examples//batch_size
#定义两个placeholder
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])
#定义神经网络中间层
W=tf.Variable(tf.zeros([784,120]))
b=tf.Variable(tf.zeros([120]))
Wx_plus_b_L1=tf.matmul(x,W)+b
L1=tf.nn.sigmoid(Wx_plus_b_L1)

# Weight_L2=tf.Variable(tf.random_normal([120,60]))
# biases_L2=tf.Variable(tf.zeros([60]))
# Wx_plus_b_L2=tf.matmul(L1,Weight_L2)+biases_L2
# L2=tf.nn.sigmoid(Wx_plus_b_L2)
#定义神经网络输出层
Weight_L2=tf.Variable(tf.random_normal([120,10]))
biases_L2=tf.Variable(tf.zeros([10]))
Wx_plus_b_L2=tf.matmul(L1,Weight_L2)+biases_L2
prediction=tf.nn.softmax(Wx_plus_b_L2)
#二次代价函数
loss=tf.reduce_mean(tf.square(y-prediction))
train_step=tf.train.GradientDescentOptimizer(0.3).minimize(loss)
#初始化变量
init=tf.global_variables_initializer()
#结果存放在一个布尔型列表中
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
#求准确率
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(200):
        for batch in range(n_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print('epoch'+str(epoch)+',testing acc'+str(acc))