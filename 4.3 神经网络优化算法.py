'''
反向传播算法和梯度下降算法调整神经网络中参数的取值。
    梯度下降算法主要用于优化单个参数的取值，而反向传播算法给出了一个高效的方式在所有参数上使用梯度下降算法
'''
import tensorflow as tf
batch_size=n
#每次读取一小部分数据作为当前的训练数据来执行反向传播算法
x=tf.placeholder(tf.float32,shape=(batch_size,2),name='x-input')
y_=tf.placeholder(tf.float32,shape=(batch_size,1),name='y-input')
#定义神经网络结构和优化算法
loss=...
train_step=tf.train.AdamOptimizer(0.001).minimize(loss)
#训练神经网络
with tf.Session() as sess:
    #参数初始化
    ...
    for i in range(STEPS):
        ...
        current_X,current_Y=...
        sess.run(train_step,feed_dict={x:current_X,y_:current_Y})