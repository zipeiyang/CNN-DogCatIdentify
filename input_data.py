import tensorflow as tf
import numpy as np
import os
# 获取文件路径和标签
def get_files(file_dir):
    # file_dir: 文件夹路径
    # return: 乱序后的图片和标签

    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    # 载入数据路径并写入标签值
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0] == 'cat':
            cats.append(file_dir + file)
            label_cats.append(0)
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1)
    print("There are %d cats\nThere are %d dogs" % (len(cats), len(dogs)))

    # 打乱文件顺序
    image_list = np.hstack((cats, dogs))   #a=[1,2,3] b=[4,5,6] print(np.hstack((a,b)))
                                           #输出：[1 2 3 4 5 6 ]
    label_list = np.hstack((label_cats, label_dogs))
    temp = np.array([image_list, label_list])
    temp = temp.transpose()     # 转置
    np.random.shuffle(temp)     ##利用shuffle打乱顺序

    ##从打乱的temp中再取出list（img和lab）
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]  #字符串类型转换为int类型

    return image_list, label_list
# 生成相同大小的批次
def get_batch(image, label, image_W, image_H, batch_size, capacity):
    # image, label: 要生成batch的图像和标签list
    # image_W, image_H: 图片的宽高
    # batch_size: 每个batch有多少张图片
    # capacity: 队列容量，一个队列最大多少
    # return: 图像和标签的batch

    # 将python.list类型转换成tf能够识别的格式
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # 生成队列 ，将image 和 label 放倒队列里
    input_queue = tf.train.slice_input_producer([image, label])

    image_contents = tf.read_file(input_queue[0])  ## 读取图片的全部信息
    label = input_queue[1]
    #将图像解码，不同类型的图像不能混在一起，要么只用jpeg，要么只用png等
    ## 把图片解码，channels ＝3 为彩色图片, r，g ，b  黑白图片为 1 ，也可以理解为图片的厚度
    image = tf.image.decode_jpeg(image_contents, channels=3)

    # 统一图片大小
    # 将图片以图片中心进行裁剪或者扩充为 指定的image_W，image_H
    # image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    # 我的方法

    image = tf.image.resize_images(image, [image_H, image_W], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)     #最近邻插值方法
    image = tf.cast(image, tf.float32)          #string类型转换为float
    # image = tf.image.per_image_standardization(image)   # 对数据进行标准化,标准化，就是减
                                                           #去它的均值，除以他的方差

    # 生成批次  num_threads 有多少个线程根据电脑配置设置  capacity 队列中 最多容纳图片的个数
     #tf.train.shuffle_batch 打乱顺序，
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,   # 线程
                                              capacity=capacity)

    # 这两行多余？ 重新排列label，行数为[batch_size]，有兴趣可以试试看
    # label_batch = tf.reshape(label_batch, [batch_size])


    return image_batch, label_batch
# TEST
import matplotlib.pyplot as plt

BATCH_SIZE = 2
CAPACITY = 256
IMG_W = 208
IMG_H = 208

train_dir = "data\\train\\"
 #调用前面的两个函数，生成batch
image_list, label_list = get_files(train_dir)
image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

#开启会话session，利用tf.train.Coordinator()和tf.train.start_queue_runners(coord=coord)来监控队列（这里有个问题：官网的start_queue_runners()是有两个参数的，sess和coord，但是在这里加上sess的话会报错）。
#利用try——except——finally结构来执行队列操作（官网推荐的方法），避免程序卡死什么的。i<2执行两次队列操作，每一次取出2张图放进batch里面，然后imshow出来看看效果

with tf.Session() as sess:
    i = 0
    ##  Coordinator  和 start_queue_runners 监控 queue 的状态，不停的入队出队1
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
        while not coord.should_stop() and i < 1:
            img, label = sess.run([image_batch, label_batch])

            for j in np.arange(BATCH_SIZE):
                print("label: %d" % label[j])
                plt.imshow(img[j, :, :, :])
                plt.show()
            i += 1
   #队列中没有数据
    except tf.errors.OutOfRangeError:
        print("done!")
    finally:
        coord.request_stop()
    coord.join(threads)
