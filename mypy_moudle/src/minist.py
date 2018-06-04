import numpy as np
import tensorflow as tf

from src.gen_captcha import gen_captcha_text_and_image

MAX_CAPTCHA=1
CHAR_SET_LEN=10
def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('验证码最长4个字符')

    vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)
    def char2pos(c):
        if c =='_':
            k = 62
            return k
        k = ord(c)-48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map')
        return k
    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
    return vector
# 把彩色图像转为灰度图像（色彩对识别验证码没有什么用）
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img
IMAGE_HEIGHT=60
IMAGE_WIDTH=60
# 生成一个训练batch
def get_next_batch(batch_size=128):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT*IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA*CHAR_SET_LEN])

    # 有时生成图像大小不是(60, 160, 3)
    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = gen_captcha_text_and_image()
            if image.shape == (60, 60, 3):
                return text, image

    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        image = convert2gray(image)

        batch_x[i,:] = image.flatten() / 255 # (image.flatten()-128)/128  mean为0
        batch_y[i,:] = text2vec(text)

    return batch_x, batch_y

# mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)
# print(mnist.train.images.shape,mnist.train.labels.shape)
# print(mnist.test.images.shape,mnist.test.labels.shape)
# print(mnist.validation.images.shape,mnist.validation.labels.shape)

text, image = gen_captcha_text_and_image()
width,height,chn=image.shape
x=tf.placeholder(tf.float32,[None,width*height],name='x')
w=tf.Variable(tf.zeros([width*height,10]),name='w')
b=tf.Variable(tf.zeros([10]),name='b')
y=tf.nn.softmax(tf.matmul(x,w)+b)
y_=tf.placeholder(tf.float32,[None,10])
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    i=0
    while True:
        xb,yb=get_next_batch()
        # batch_xs,batch_ys=mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={x:xb,y_:yb})
        # print(i)
        i=i+1
        if i%50==0:
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))#argmax最大值所在的位置
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            right=sess.run(accuracy, feed_dict={x: xb, y_: yb})
            print(right)
            if right>0.5:
                break


