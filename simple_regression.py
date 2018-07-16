import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

x=np.linspace(-1,1,100)[:,np.newaxis]  # np.newaxis 为 numpy.ndarray（多维数组）增加一个轴
noise=np.random.normal(0,0.1,size=x.shape)
y=np.power(x,2)+noise   #python之numpy.power()数组元素求n次方

#plot data
plt.scatter(x,y)
plt.show()

tf_x=tf.placeholder(tf.float32,x.shape)
tf_y=tf.placeholder(tf.float32,y.shape)

#neural network layers
l1=tf.layers.dense(tf_x,10,tf.nn.relu)
output=tf.layers.dense(l1,1)

loss=tf.losses.mean_squared_error(tf_y,output)
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.5)
train_op=optimizer.minimize(loss)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

plt.ion()   #打开交互模式

for step in range(100):
    #train and net output
    _,l,pred=sess.run([train_op,loss,output],feed_dict={tf_x:x,tf_y:y})
    if step % 5==0:
        plt.cla()   #清除轴，当前活动轴在当前图中。它保持其他轴不变
        plt.scatter(x,y)
        plt.plot(x,pred,'r-',lw=5)
        plt.text(0.5,0,'Loss=%.4f'%l,fontdict={'size':20,'color':'red'})
        plt.pause(0.1)
plt.ioff()  #关闭交互模式
plt.show()
