'''
tensorflow首先要定义神经网络的结构，然后再把数据放入结构当中取运算和training
因为tensorflow是采用数据流图来计算，所以首先我们得创建一个数据流流图，然后再将我们的数据
（数据以张量（tensor）的形式存在）放在数据流图中计算
'''
import tensorflow as tf
import numpy as np
x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*0.1+0.3

#create tensorflow structure start#
Weights=tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases=tf.Variable(tf.zeros([1]))

y=Weights*x_data+biases
loss=tf.reduce_mean(tf.square(y-y_data))
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss)

init=tf.initialize_all_variables()

#create tensorflow structure end#
sess=tf.Session()
sess.run(init)  #very important

for step in range(201):
    sess.run(train)
    if step%20==0:
        print(step,sess.run(Weights),sess.run(biases))

