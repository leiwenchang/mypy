import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import NormalNN as NN
from mpl_toolkits.mplot3d import Axes3D
# !/usr/bin/python
# -*- coding: UTF-8 -*-
# 文件名：test.py
from numpy.core.tests.test_mem_overlap import xrange

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def liner():
    real_w = [0.1] #实际权重
    dim = real_w.__len__()
    real_b = 0.3 #实际偏移
    if dim != real_w.__len__():  # __len__是数组的长度
        print(real_w.__len__())

    x_data = np.float32(np.random.rand(dim, 200))  # 随机输入
    y_data = np.dot(real_w, x_data) + real_b

    # 构造一个线性模型
    #
    b = tf.Variable(tf.zeros([dim]))
    W = tf.Variable(tf.random_uniform([1, dim], -1.0, 1.0))
    y = tf.matmul(W, x_data) + b

    # 最小化方差
    loss = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)

    # 初始化变量
    init = tf.initialize_all_variables()

    # 启动图 (graph)
    sess = tf.Session()
    sess.run(init)

    lossArray = {}
    # 拟合平面
    for step in xrange(0, 201):
        sess.run(train)
        if step % 20 == 0:
            lossArray[step] = (sess.run(loss))
            print(step, sess.run(W), sess.run(b), sess.run(loss))
    print(lossArray.get(0))
    # plt.plot(lossArray.values())
    # plt.show()


def plot():
    labels = 'frogs', 'hogs', 'dogs', 'logs'
    sizes = 15, 20, 45, 10
    colors = 'yellowgreen', 'gold', 'lightskyblue', 'lightcoral'
    explode = 0, 0.1, 0, 0
    # plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=50)
    plt.plot(sizes)
    plt.axis('equal')
    plt.show()


def test():
    sess = tf.Session()
    a = [[1], [1]]
    b = [[1]]
    d = tf.reshape(a, [1, 2])
    c = tf.matmul(b, d)
    print(c.get_shape().as_list())
    print(sess.run(c))


def place_test():
    # 定义‘符号’变量，也称为占位符
    a = tf.placeholder("float", [2, 2])
    b = tf.placeholder("float", [2, 2])

    y = tf.matmul(a, b)  # 构造一个op节点

    sess = tf.Session()  # 建立会话
    # 运行会话，输入数据，并计算节点，同时打印结果
    print(sess.run(y, feed_dict={a: [[3, 3], [3, 3]], b: [[3, 3], [3, 3]]}))
    # 任务完成, 关闭会话.
    sess.close()


def test1():
    tf.InteractiveSession()
    a = tf.Variable([[1], [1]])
    d = tf.Variable([[1], [1]])
    b = tf.constant([[1]])
    e = tf.add(a, d)
    c = tf.multiply(e, b)
    a.initializer.run()
    d.initializer.run()
    print(c.eval())


def trustNN():
    fig = plt.figure()
    ax = Axes3D(fig)
    dim = 2
    size=20
    lb=1
    ub=10

    a=np.random.randint(1,10,size=[size,dim,dim])
    b=np.random.randint(1,10,size=[size,dim])
    ra=a.reshape((2,size,dim))
    x=ra[0]
    y=ra[1]
    c=np.linalg.solve(a,b)
    # print(x)
    ax = plt.subplot(111, projection='3d')
    ax.scatter(x,y,c)
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()
    # c=np.zeros((1,10),int)[0]
    # print(c)
    # print(x_data )
    # for x in range(10) :
        # print(x_data[x:x+dim-1][0])
        # a=x_data[x:x+2]
        # b=y_data[x]
        # c[x]=np.linalg.solve(a,b)
        # print(c)

def plot3D():
    data = np.random.randint(0, 255, size=[3, 2, 2])

    x, y, z = data[0], data[1], data[2]
    # print(x)
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    #  将数据点分成三部分画，在颜色上有区分度
    print(x[:10])
    print(y[:10])
    print(z[:10])
    ax.scatter(x[:2], y[:2], z[:2], c='y')  # 绘制数据点
    x1=[[1,11],[2,2]]
    y1=[[101,131],[142,162]]
    z1=[[201,201],[202,202]]
    print(z1)
    #
    ax.scatter(x1, y1, z1, c='r')  # 绘制数据点
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()


def arrayTest():
    data = np.random.randint(0, 255, size=[40, 2, 2])
    # print(data)
    print(data[0][0][0])

def luanqibaz():
    x_data=np.random.randint(0, 2, [100, 16], dtype=int)
    low=np.matmul([[1,2,4,8,16,32,64,128]],x_data.transpose()[0:8,:])
    high=np.matmul([[1,2,4,8,16,32,64,128]],x_data.transpose()[8:16,:])
    real_y=low+high
    sess=tf.Session
    print(low)
    print(high)




if __name__ == '__main__':
    NN.NN()
    # luanqibaz()
    # liner()
    # test1()
    # trustNN()
    # plot3D()
    # plot()
    # arrayTest()
