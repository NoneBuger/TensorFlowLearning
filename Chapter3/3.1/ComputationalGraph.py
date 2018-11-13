#coding:utf-8

import tensorflow as tf

a = tf.constant([2.4,5.2],name="a")
b = tf.constant([3.4,5.9],name="b")
result = a + b
sess = tf.Session()
print(sess.run(result))

