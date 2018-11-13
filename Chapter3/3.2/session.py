'''
Mode 1:
# create a session
sess = tf.Session()
# use this session to get the result,for example: sess.run(result)
sess.run(...)
#close the session and release the resource
sess.close()
'''

# Mode 2:
import tensorflow as tf
import example

result = example.TensorAdd()

# create a session, and manage this session with the Python Context Manager
with tf.Session() as sess:
	print(sess.run(result))
# do not need call Session.close() to close this session
# session will close and release resources while exit the context

if __name__ == "__main__":
	pass


