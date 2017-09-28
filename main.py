import tensorflow as tf
import argparse
import os
import sys
from tensorflow.examples.tutorials.mnist import input_data
from utils import *

data_path  = "data/"
model_dir  = "model/"
saved_model_path = os.path.join(model_dir,"model.ckpt")

sample_size = 28
default_sample_upsacle = 7

N0 = 784
N1 = 128
N2 = 2
init = None

def leakyRelu(x):
	return tf.maximum(0.2*x,x)

def encoder(x,reuse):

	nl = leakyRelu
	x = tf.layers.dense(x, N1, activation=nl  , kernel_initializer = init, name="e1", reuse=reuse)
	x = tf.layers.dense(x, N2, activation=None, kernel_initializer = init, name="e2", reuse=reuse)
	return x

def decoder(x,reuse):

	nl = leakyRelu
	x = tf.layers.dense(x, N1, activation=nl  , kernel_initializer = init, name="d2", reuse=reuse)
	x = tf.layers.dense(x, N0, activation=None, kernel_initializer = init, name="d3", reuse=reuse)
	return tf.nn.sigmoid(x)

def main(args):

	input  = tf.placeholder(tf.float32,shape = (None,N0))
	latent = tf.placeholder(tf.float32,shape = (None,N2))
	encode = encoder(input ,None)
	decode = decoder(latent,None)
	out    = decoder(encoder(input,True),True)
	mnist  = input_data.read_data_sets(data_path, one_hot=True)

	if not os.path.exists(model_dir):
		os.makedirs(model_dir)
	if not os.path.exists(data_path):
		os.makedirs(data_path)

	if args.verbose:
		print_var_list("e")
		print_var_list("d")
	else:
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

	if args.train:
		train_model(mnist,input,out)
		return

	if args.linspace:
		linspace(mnist,encode,decode,input,latent)
		return

	if args.random:
		random_latent(mnist,encode,decode,input,latent)
		return

	if args.grid:
		draw_2d_grid(mnist,encode,decode,input,latent)
		return

def train_model(mnist,input,out):

	disp_sample = 4
	lr = 5e-4
	batch = 256
	training_batches = 20000
	loss_checkpoint = 200
	display_checkpoint = 20

	loss = tf.reduce_mean(tf.square(input-out),[0,1])
	opt  = tf.train.AdamOptimizer(lr).minimize(loss)

	with tf.Session() as sess:

		saver = tf.train.Saver()
		sess.run(tf.global_variables_initializer())

		for i in range(training_batches):

			x,_ =  mnist.train.next_batch(batch)
			sess.run(opt,feed_dict = {input:x})

			if (i%loss_checkpoint == 0):

				x_,_ = mnist.test.next_batch(64)
				samples = []
				for j in range(disp_sample):
						samples.append(x_[j:j+1])
				cur_loss = sess.run(loss,feed_dict = {input:x_})
				print("Iteration: " + str(i) + " Loss " + str(cur_loss))
				saver.save(sess,saved_model_path)

			if (i%display_checkpoint == 0):
				for i,sample in enumerate(samples):
					disp(sess,out,input,sample,i)

def draw_2d_grid(mnist,encode,decode,input,latent):

	latent_dim = encode.get_shape().as_list()[-1]
	if (latent_dim != 2):
		print("Error - the latent space is not 2d. is " + str(latent_dim))
		return

	grid = 30
	std_offsets = 1.5
	img_size = sample_size*grid	

	with load_session() as sess:
		mean_, var_ = latent_space_moments(sess,mnist,encode,input)
		std = np.sqrt(var_)

		start = mean_ + std_offsets * std
		end   = mean_ - std_offsets * std
		img = np.zeros((img_size,img_size))

		for x in range(grid):
			for y in range(grid):

				alpha_x = x/grid
				alpha_y = y/grid
				x_ = start*alpha_x + end*(1-alpha_x)   
				y_ = start*alpha_y + end*(1-alpha_y)   
				cur_latent = np.array([[x_,y_]])
				decoded = sess.run(decode,feed_dict = {latent:cur_latent})
				img[x*sample_size:(x+1)*sample_size,y*sample_size:(y+1)*sample_size] = np.reshape(decoded,[sample_size,sample_size])

		cv2.imshow("grid" ,show(img,f=1,xy=img_size))
		cv2.waitKey(0)	

def random_latent(mnist,encode,decode,input,latent):

	samples_num = 10
	latent_dim = encode.get_shape().as_list()[-1]

	with load_session() as sess:
		mean_, var_ = latent_space_moments(sess,mnist,encode,input)

		for _ in range(samples_num):		
			random_latent = np.random.normal(mean_, np.sqrt(var_), (1,latent_dim))
			decoded = sess.run(decode,feed_dict = {latent:random_latent})
			cv2.imshow("random" ,show(decoded))
			cv2.waitKey(1000)

def linspace(mnist,encode,decode,input,latent):

	steps = 100
	with load_session() as sess:

		x,_ =  mnist.train.next_batch(2)
		cv2.imshow("start",show(x[0:1]))
		cv2.imshow("end",show(x[1:2]))
		y = sess.run(encode,feed_dict = {input:x})
		for i in range(steps+1):

			alpha = i/steps
			mid_y =  (1-alpha)*y[0:1,:] + alpha*y[1:2,:]
			decoded = sess.run(decode,feed_dict = {latent:mid_y})

			cv2.imshow("mid" ,show(decoded))
			cv2.waitKey(100)

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--n', '-n', type=int, default=3)
	parser.add_argument('--verbose' , action='store_true',	default=False)
	parser.add_argument('--linspace', action='store_true',	default=False)
	parser.add_argument('--random'  , action='store_true',	default=False)
	parser.add_argument('--grid'	, action='store_true',	default=False)
	parser.add_argument('--train'   , action='store_true',	default=False)
	args = parser.parse_args()
	main(args)