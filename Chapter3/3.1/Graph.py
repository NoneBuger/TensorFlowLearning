#coding:utf-8

import tensorflow as tf

g1 = tf.Graph()
with g1.as_default():
	# define v in g1, and init v as 0
	v = tf.get_variable("v",initializer=tf.zeros_initializer(shape=[1]))

g2 = tf.Graph()
with g2.as_default():
	# define v in g2, and init v as 0
	v = tf.get_variable("v",initializer=tf.ones_initializer(shape=[1]))

#read the value of 'v' in graph g1
with tf.Session(graph = g1) as sess:
	tf.initialize_all_variables().run()
	with tf.variable_scope("",reuse=True):
		print(sess.run(tf.get_variable("v")))
		
#read the value of 'v' in graph g2
with tf.Session(graph = g2) as sess:
	tf.initialize_all_variables().run()
	with tf.variable_scope("",reuse=True):
		print(sess.run(tf.get_variable("v")))
