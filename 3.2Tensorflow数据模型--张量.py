'''
在Tensorflow程序中，所有的数据都通过张量的形式来表示。从功能的角度上看，张量可以被简单理解为多维数组。
如下代码，并不会得到加法的结果，而会得到对结果的一个引用
'''
import tensorflow as tf
a=tf.constant([1.0,2.0],name='a')
b=tf.constant([3.0,2.0],name='b')
result=tf.add(a,b,name='add')
print(result)
'''
输出：
    Tensor("add:0", shape=(2,), dtype=float32)
'''
'''
从上面的代码可以看出Tensorflow中的张量和Numpy中的数组不同，Tensorflow计算的结果不是一个具体的数字，而是一个张量的结构。
一个张量中主要保存了三个属性：名字（name）、维度（shape）和类型（type）
每一个张量会有一个唯一的类型。Tensorflow会对参与运算的所有张量进行类型的检查，当发现类型不匹配时会报错
'''