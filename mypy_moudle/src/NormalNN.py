import tensorflow as tf
import numpy as np
from numpy.core.tests.test_mem_overlap import xrange
def NN():
    INPUT_SIZE=16
    HIDE_LAYER_SIZE1=100
    HIDE_LAYER_SIZE2 = 100
    OUTPUT_SIZE=8

    baseCode=[[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]]

    x_data = np.float32(np.random.randint(0, 2, [100000, 16]))
    low = np.matmul(baseCode, x_data.transpose()[0:8, :])
    high = np.matmul(baseCode, x_data.transpose()[8:16, :])
    y_data = low + high

    w1=tf.Variable(tf.random_normal([INPUT_SIZE,HIDE_LAYER_SIZE1]))
    b1=tf.Variable(tf.zeros([HIDE_LAYER_SIZE1]))
    w2=tf.Variable(tf.random_normal([HIDE_LAYER_SIZE1,HIDE_LAYER_SIZE2]))
    b2=tf.Variable(tf.zeros([HIDE_LAYER_SIZE2]))
    w_out=tf.Variable(tf.random_normal([HIDE_LAYER_SIZE2,OUTPUT_SIZE]))
    b3=tf.Variable(tf.zeros([OUTPUT_SIZE]))
    # x=tf.placeholder(dtype="float",shape=[100,INPUT_SIZE],name="input")
    x=x_data
    a=tf.nn.sigmoid(tf.matmul(x,w1)+b1)
    b=tf.nn.relu(tf.matmul(a,w2)+b2)
    y=tf.nn.sigmoid(tf.matmul(b,w_out)+b3)
    real_y=tf.matmul(baseCode,tf.transpose(y))
    # 最小化方差
    loss = tf.reduce_mean(tf.square(real_y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)

    # 初始化变量
    init = tf.initialize_all_variables()

    # 启动图 (graph)
    sess = tf.Session()
    sess.run(init)
    # for step in xrange(0, 201):
    sess.run(train)
    print(sess.run(loss))

