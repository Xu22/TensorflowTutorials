'''
会话拥有并管理Tensorflow程序运行时的所有资源。当所有计算完成之后需要关闭会话来帮助系统回收资源，
否则就可能出现资源泄露的问题。
Tensorflow中使用会话的模式一般有两种，第一种模式需要明确调用会话生成函数和关闭会话函数
        流程如下：
        sess=tf.Session()   #创建一个会话
        sess.run(...)   #得到张量的取值
        sess.close()    #关闭会话使得本次运行中使用到的资源可以被释放
    这种模式，在当程序因为异常而退出时，关闭会话的函数可能就不会被执行从而导致资源泄漏。
    为了解决异常退出时资源释放的问题，Tensorflow可以通过python的上下文管理器来使用会话
        流程如下：
        with tf.Session() as sess:
            sess.run(....)
Tensorflow会自动生成一个默认的计算图，如果没有特殊指定，运算会自动加入这个计算图中。
但Tensorflow不会自动生成默认的会话，而是需要手动指定。当默认的会话被指定之后可以通过tf.Tensor.eval
函数来计算一个张量的取值。如下：
        sess=tf.Session()
        with sess.as_default():
            print(result.eval())
    在交互式环境下，tf.InteractiveSession()函数会自动将生成的会话注册为默认会话。如下：
        sess=tf.InteractiveSession()
        print(result.eval())
        sess.close()
通过tf.InteractiveSession函数可以省去将产生的会话注册为默认会话的过程。无论使用哪种方法都可以通过
ConfigProto Protocol Buffer来配置需要生成的会话。如下：
        config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
        sess1=tf.InteractiveSession(config=config)
        sess2=tf.Session(config=config)
    通过ConfigProto可以配置类似并行的线程数、GPU分配策略、运算超时时间等参数。
    在这些参数中，最常使用的有两个。第一个allow_soft_placement,这是一个布尔型的参数
'''