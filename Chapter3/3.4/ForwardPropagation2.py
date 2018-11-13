#coding:utf-8

import tensorflow as tf
from numpy.random import RandomState

# define training data bacth size:
batch_size = 8

# weight 1
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1),name="w1")
	
# weight 2
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1),name="w2")

# input layer
x = tf.placeholder(tf.float32,shape=(None,2),name='x-input')
y_ = tf.placeholder(tf.float32,shape=(None,1),name='y-input')

a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

# define loss func and BackPropagation Algrithm
cross_entropy = -tf.reduce_mean(
	y_ * tf.log(tf.clip_by_value(y,1e-10,1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# product an random dataset
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)

Y = [[int(x1+x2 < 1)] for (x1,x2) in X ]


with tf.Session() as sess:
	init_op = tf.initialize_all_variables()
	#sess.run(w1.initializer)
	sess.run(init_op)
	print("w1:",sess.run(w1))
	print("w2:",sess.run(w2))

# define training times
	STEPS = 5000
	for i in range(STEPS):
		# select batch_size samples to train per time
		start = (i * batch_size) % dataset_size
		end = min(start+batch_size,dataset_size)
		
		# update paraments
		sess.run(train_step,
			feed_dict={x:X[start:end],y_:Y[start:end]})
		if i % 1000 == 0:
			total_cross_entropy = sess.run(
				cross_entropy,feed_dict={x:X,y_:Y})
			print("After %d Training step(s),cross entropy on all data is %g"
				%(i,total_cross_entropy))	
	
		
	print("w1:",sess.run(w1))
	print("w2:",sess.run(w2))

