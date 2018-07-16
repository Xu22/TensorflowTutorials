import tensorflow as tf

m1=tf.constant([[2,2]])
m2=tf.constant([[3],[3]])

dot_operation=tf.matmul(m1,m2)
# print(dot_operation)    #wrong! no result

#method 1 use Session
sess=tf.Session()
result=sess.run(dot_operation)
print(result)
sess.close()

#method 2 use Session
with tf.Session() as sess:
    result=sess.run(dot_operation)
    print(result)
