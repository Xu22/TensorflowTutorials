'''
tensorflow的特点：
    使用图（graph）来表示计算任务
    在被称为 会话（session）的上下文（context）中执行图
    使用tensor表示数据
    通过变量（variable）维护状态
    使用feed和fetch可以为任意的操作（arbitrary operation）赋值或者从其中获取数据

'''
import tensorflow as tf

#创建一个常量节点op
#加到默认图中
#构造器的返回值代表该常量op的返回值
matrix1=tf.constant([[3.,3.]])

#创建另一常量op，产生一个2*1矩阵
matrix2=tf.constant([[2.],[2.]])

#创建一个矩阵乘法matmul op,把‘matrix1’和‘matrix2’作为输入
#返回值product 代表矩阵乘法的结果
product=tf.matmul(matrix1,matrix2)

'''
默认图中现在有三个节点，两个constant（） op,和一个matmul() op.为了真正
进行矩阵乘法运算，并得到矩阵乘法结果，必须在会话里启动这个图

'''
#启动默认图
sess=tf.Session()
result=sess.run(product)
print(result)
sess.close()    #任务完成，关闭会话

'''
Session对象在使用完成后需要关闭以释放资源，除了显式调用close外，也可以使用‘with’代码块
来自动完成关闭动作
'''
with tf.Session() as sess:
    result=sess.run([product])
    print(result)

#变量
#创建一个变量，初始化为标量0
state=tf.Variable(0,name='counter')
#创建一个op,其作用是使state增加1
one=tf.constant(1)
new_value=tf.add(state,one)
update=tf.assign(state,new_value)

#启动图后，变量必须先经过 初始化（init）op
#首先必须增加一个初始化op到途中
init_op=tf.initialize_all_variables()
with tf.Session() as sess:
    #运行init op
    sess.run(init_op)
    print(sess.run(state))
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

'''
feed ,该机制可以临时替代图中的任意操作中的tensor
feed只在调用它的方法内有效，方法结束，feed就会消失。
最常见的用例是将某些特殊的操作指定‘feed’操作，标记的方法是使用tf.placeholder()为这些操作创建占位符
'''
input1=tf.placeholder(tf.float32)
input2=tf.placeholder(tf.float32)
output=tf.multiply(input1,input2)
with tf.Session() as sess:
    print(sess.run([output],feed_dict={input1:[7.],input2:[2.]}))