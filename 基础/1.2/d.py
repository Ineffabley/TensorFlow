import  tensorflow as tf
#迭代
state=tf.Variable(0)
n=tf.add(state,tf.constant(1))
#state的值变为n
update=tf.assign(state,n)
with tf.Session() as sess:
    # 必须要使用global_variables_initializer的场合
    # 含有tf.Variable的环境下，因为tf中建立的变量是没有初始化的，也就是在debug时还不是一个tensor量，而是一个Variable变量类型
    sess.run(tf.global_variables_initializer())
    print(sess.run(state))  #state初始值为0
    for _ in range(3):
        #print(sess.run(n))

        sess.run(update)   #运行三次add和assign
        #print(sess.run(n))
        print(sess.run(state))