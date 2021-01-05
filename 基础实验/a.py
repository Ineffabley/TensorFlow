import tensorflow as  tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 设置当前使用的GPU设备仅为0号设备  设备名称为'/gpu:0'
hello = tf.constant("Hello,TensorFlow")
# 创建一个Session会话对象，设置定量的GPU使用量，会话封装了Tensorflow运行时的状态和控制。
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 占用GPU90%的显存
sess = tf.Session(config=config)
print(sess.run(hello))