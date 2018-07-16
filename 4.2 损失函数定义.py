'''
神经网络模型的效果以及优化的目标是通过损失函数来定义的
    交叉熵（cross entropy）是常用的评判方法之一。交叉熵刻画了两个概率分布之间的距离
            H(p,q)=-sum(p(x)log(q(x)))
        它刻画的是通过概率分布q来表达概率分布p的困难程度，p代表的是正确答案，q代表的是预测值。
        交叉熵值越小，两个概率分布越接近
'''
import tensorflow as tf

#交叉熵
# tf.clip_by_value函数可以将一个张量中的数值限制在一个范围内，这样可以避免一些运算错误（比如log0是无效的）
# cross_entropy=-tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))

'''
4.2.2 自定义损失函数
'''
