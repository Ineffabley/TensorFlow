import tensorflow as tf

a = 3
# Create a variable
w = tf.Variable([[20, 10]])    #一行两列
x = tf.Variable([[1], [2]])    #两行一列
y = tf.matmul(w, x)
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    print(y.eval())     #一行一列
