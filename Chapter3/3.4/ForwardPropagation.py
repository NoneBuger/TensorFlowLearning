#coding:utf-8

import tensorflow as tf

# input layer
x = tf.constant([[0.7,0.9]],name="input")

# weight 1
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1),name="w1")
	
# weight 2
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1),name="w2")

a = tf.matmul(x,w1)
y = tf.matmul(a,w2)


with tf.Session() as sess:
	init_op = tf.initialize_all_variables()
	#sess.run(w1.initializer)
	sess.run(init_op)
	output = sess.run(y)
	if float(output[0][0]) > 0 :
		print("this product is up to standard!")
	else:
		print("this product is not up to standard!")



