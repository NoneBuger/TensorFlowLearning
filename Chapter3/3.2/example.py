import tensorflow as tf
'''
tf.constant is a calculation,the result of the calculation is a tensor,save in
'''
def TensorAdd():
	a = tf.constant([2.3,4.4],name="a")
	b = tf.constant([4.2,3.1],name="b")
	result = tf.add(a,b,name="add")
	return result
if __name__ == "__main__":
	print(TensorAdd())
'''
Output:
Tensor("add:0",shape=(2,),dtype=float32)
'''
