'''
4.4.1 学习率的设置
4.4.2 过拟合问题
    加入正则化
    w=tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
    y=tf.matmul(x,w)
    loss=tf.reduce_mean(tf.square(y_-y))+tf.contrib.layers.l2_regularizer(lambda)(w)

4.4.3 滑动平均模型

'''