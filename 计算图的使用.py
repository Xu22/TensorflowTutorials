'''
Tensorflow中的计算图不仅仅可以用来隔离张量和计算，它还提供了管理张量和计算的机制。
计算图可以通过tf.Graph.device函数来指定运行计算的设备。这为Tensorflow使用GPU提供了机制

'''
import tensorflow as tf

a=tf.constant([1,2],name='a')
b=tf.constant([2,3],name='b')

result=a+b
'''
在这个过程中，Tensorflow会自动将定义的计算转化为计算图上的节点。
在Tensorflow程序中，系统会自动维护一个默认的计算图，通过tf.get_default_graph函数
可以获取当前默认的计算图
除了使用默认的计算图，Tensorflow支持通过tf.Graph函数来生成新的计算图。不同计算图上的张量和运算都不会共享
'''
print(a.graph is tf.get_default_graph())

g1=tf.Graph()
with g1.as_default():
    #在计算图g1中定义变量‘v',并设置初始值为0
    v=tf.get_variable('v',initializer=tf.zeros_initializer,shape=[1])

g2=tf.Graph()
with g2.as_default():
    #在计算图g2中定义变量‘v’，并设置初始值为1
    v=tf.get_variable('v',initializer=tf.ones_initializer,shape=[1])

#在计算图g1中读取变量‘v’的值
with tf.Session(graph=g1) as sess:
    tf.initialize_all_variables().run()
    with tf.variable_scope('',reuse=True):
        #在计算图g1中，变量‘v’的取值应该为0，所以下面这行会输出[0.]
        print(sess.run(tf.get_variable('v')))

#在计算图g2中读取变量‘v’的值
with tf.Session(graph=g2) as sess:
    tf.initialize_all_variables().run()
    with tf.variable_scope('',reuse=True):
        #在计算图g2中，变量‘v’的取值应该为1，所以下面这行会输出[1.]
        print(sess.run(tf.get_variable('v')))

g=tf.Graph()
#指定计算运行的设备
with g.device('/gpu:0'):
    result=a+b
