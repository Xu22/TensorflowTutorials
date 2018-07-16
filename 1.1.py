'''
使用图（graphs）来表示计算任务
在被称为会话（Session）的上下文中执行图
使用tensor表示数据
通过变量（Variable）维护状态
使用feed和fetch可以为任意的操作赋值或者从其中获取数据

'''

import tensorflow as tf
import numpy as np
x_data=np.float32(np.random.rand(2,100))
y_data=np.dot([0.100,0.200],x_data)+0.300
#构造线性模型
b=tf.Variable(tf.zeros([1]))
W=tf.Variable(tf.random_uniform([1,2],-1.0,1.0))
y=tf.matmul(W,x_data)+b
# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化变量
init = tf.initialize_all_variables()

# 启动图 (graph)
sess = tf.Session()
sess.run(init)

# 拟合平面
for step in range(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))