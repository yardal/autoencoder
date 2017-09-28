import functools
import operator
import numpy as np
import cv2
import tensorflow as tf
from main import *

def latent_space_moments(sess,mnist,encode,input):

	batch_size = 128
	x,_ =  mnist.train.next_batch(batch_size)
	mean, var = tf.nn.moments(encode, axes=[0,1])
	mean_, var_ = sess.run([mean, var],feed_dict = {input:x})

	return mean_,var_

def load_session():

	try:
		sess = tf.Session()
		saver = tf.train.Saver()
		saver.restore(sess, saved_model_path)
		return sess

	except:
		print("Unable to load model from:\t" + str(model_dir))
		print("Run with --train")
		sys.exit(0)

def show(im, f=default_sample_upsacle, xy=sample_size):

	im = np.reshape(im,[xy,xy])
	im = np.kron(im,np.ones((f,f)))
	return im

def disp(sess,out,input,sample,idx):

	f = 5
	decoded = sess.run(out,feed_dict = {input:sample})
	x_im       = np.reshape(sample ,[sample_size,sample_size])
	decoded_im = np.reshape(decoded,[sample_size,sample_size])			
	cv2.imshow("IN"+str(idx) ,np.kron(x_im,np.ones((f,f))))
	cv2.imshow("OUT"+str(idx),np.kron(decoded_im,np.ones((f,f))))

	size = 180
	cv2.moveWindow("IN"+str(idx),idx*(size-20),0)
	cv2.moveWindow("OUT"+str(idx),idx*(size-20),size)
	cv2.waitKey(1)

def print_var_list(prefix):

	var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,prefix)
	total = 0
	for i in var_list:
		num_param_in_var = functools.reduce(operator.mul,i.get_shape().as_list())
		strr = i.name + "\tParams: " + str(num_param_in_var)
		print(strr.expandtabs(27))
		total = total + num_param_in_var
	print("Total: " + str(total))