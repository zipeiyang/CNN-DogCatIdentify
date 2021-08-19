import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import input_data
import numpy as np
import model
import os
#从指定目录中选取一张图片
def get_one_image(train):
    files = os.listdir(train)
    n = len(files)
    n = len(train)
    ind = np.random.randint(0,n)
    img_dir = os.path.join(train,files[ind])
    image = Image.open(img_dir)
    plt.imshow(image)
    plt.show()
    image = image.resize([208, 208])
    image = np.array(image)
    return image
#测试图片
def evaluate_one_image():
    #存放我们想测试的图片集
    train = './data/test/'
    image_array = get_one_image(train)
with tf.Graph().as_default():
       BATCH_SIZE = 1   # 因为只读取一副图片 所以batch 设置为1
       N_CLASSES = 2    ## 2个输出神经元，［1，0］ 或者 ［0，1］猫和狗的概率
       # 转化图片格式
       image = tf.cast(image_array, tf.float32)
       # 图片标准化
       image = tf.image.per_image_standardization(image)
       # 图片原来是三维的 [208, 208, 3] 重新定义图片形状 改为一个4D  四维的 tensor
       image = tf.reshape(image, [1, 208, 208, 3])
       logit = model.inference(image, BATCH_SIZE, N_CLASSES)
       # 因为 inference 的返回没有用激活函数，所以在这里对结果用softmax 激活
       logit = tf.nn.softmax(logit)
       # 用最原始的输入数据的方式向模型输入数据 placeholder
       x = tf.placeholder(tf.float32, shape=[208, 208, 3])

       # 我门存放模型的路径
       logs_train_dir = './Logs/train'

        # 定义saver
       saver = tf.train.Saver()

       with tf.Session() as sess:

           print("Reading checkpoints...")
           # 将模型加载到sess 中
           ckpt = tf.train.get_checkpoint_state(logs_train_dir)
           if ckpt and ckpt.model_checkpoint_path:
               global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
               saver.restore(sess, ckpt.model_checkpoint_path)
               print('Loading success, global_step is %s' % global_step)
           else:
               print('No checkpoint file found')

           prediction = sess.run(logit, feed_dict={x: image_array})
           max_index = np.argmax(prediction)
           if max_index==0:
               print('This is a cat with possibility %.6f' %prediction[:, 0])
           else:
               print('This is a dog with possibility %.6f' %prediction[:, 1])
