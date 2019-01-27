import tensorflow as tf
import tensorflow as tf2
import os  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np


x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1+0.3


w = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.zeros([1]))
loss = tf.reduce_mean(tf.square(   w*x_data+b   -   y_data   ))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train =  optimizer.minimize(loss)


init = tf.global_variables_initializer()


sess = tf.Session()
sess.run(init)
for step in range(201):
    sess.run(train)
    if step%20 ==0:
       print(step,sess.run(w),sess.run(b))


train_writer = tf.summary.FileWriter('/home/shayne/tfboard_Test', sess.graph)
train_writer.close()  #然後打開cmd  輸入 tensorboard --logdir=/home/shayne/tfboard_Test  拜訪網址 localhost:6006   之後cmd留著不要關就會自動傳圖了



state = tf.Variable(0,name='counter')
print(state.name)
one = tf.constant(1)

new_value = tf.add(state,one)
update = tf.assign(state, new_value)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))


