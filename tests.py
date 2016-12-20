from three_layer_network import build_model, relu, relu_derivative, feed_forward, \
								calculate_loss, backprop, train
#from three_layer_network import *
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np

"""
to reproduce tests, modify the three_layer_network.py file by commenting out 
'while done == True', and uncommenting 'while i < 150', and then by changing 
'if i % 1000 == 0' to 'if i % 150 == 0'
"""


def num_observations():
	obs_values = [10, 100, 1000]
	nn_input_dim = 2 # input layer dimensionality
	nn_output_dim = 2 # output layer dimensionality 
	learning_rate = 0.01 # learning rate for gradient descent
	reg_lambda = 0.01 # regularization strength
	losses_store = []
	for i in obs_values:
		X, y = datasets.make_moons(i, noise=0.1)
		num_examples = len(X) # training set size
		model = build_model(X,32,2)
		model, losses = train(model,X, y, reg_lambda=reg_lambda, learning_rate=learning_rate)
		losses_store.append(losses)
		print losses
	x = np.linspace(0,145,30)
	for i in range(len(losses_store)):
		lab = 'n_observations = ' + str(obs_values[i])
		plt.plot(x,losses_store[i],label=lab)
	plt.legend()
	plt.show()

def noise():
	noise_values = [0.01, 0.1, 0.2, 0.3, 0.4]
	nn_input_dim = 2 # input layer dimensionality
	nn_output_dim = 2 # output layer dimensionality 
	learning_rate = 0.01 # learning rate for gradient descent
	reg_lambda = 0.01 # regularization strength
	losses_store = []
	for i in noise_values:
		X, y = datasets.make_moons(200, noise=i)
		num_examples = len(X) # training set size
		model = build_model(X,32,2)
		model, losses = train(model,X, y, reg_lambda=reg_lambda, learning_rate=learning_rate)
		losses_store.append(losses)
		print losses
	x = np.linspace(0,145,30)
	for i in range(len(losses_store)):
		lab = 'noise_value = ' + str(noise_values[i])
		plt.plot(x,losses_store[i],label=lab)
	plt.legend()
	plt.show()

def reg():
	reg_values = [0.00, 0.01, 0.1, 0.2, 0.3]
	nn_input_dim = 2 # input layer dimensionality
	nn_output_dim = 2 # output layer dimensionality 
	learning_rate = 0.01 # learning rate for gradient descent
	losses_store = []
	for i in reg_values:
		reg_lambda = i # regularization strength
		X, y = datasets.make_moons(200, noise=0.2)
		num_examples = len(X) # training set size
		model = build_model(X,32,2)
		model, losses = train(model,X, y, reg_lambda=reg_lambda, learning_rate=learning_rate)
		losses_store.append(losses)
		print losses
	x = np.linspace(0,145,30)
	for i in range(len(losses_store)):
		lab = 'regularization_value = ' + str(reg_values[i])
		plt.plot(x,losses_store[i],label=lab)
	plt.legend()
	plt.show()


def lr():
	lr_values = [0.001, 0.01, 0.05]
	nn_input_dim = 2 # input layer dimensionality
	nn_output_dim = 2 # output layer dimensionality 
	reg_lambda = .01 # regularization strength
	losses_store = []
	for i in lr_values:
		learning_rate = i
		X, y = datasets.make_moons(200, noise=0.2)
		num_examples = len(X) # training set size
		model = build_model(X,32,2)
		model, losses = train(model,X, y, reg_lambda=reg_lambda, learning_rate=learning_rate)
		losses_store.append(losses)
		print losses
	x = np.linspace(0,145,30)
	for i in range(len(losses_store)):
		lab = 'learning rate = ' + str(lr_values[i])
		plt.plot(x,losses_store[i],label=lab)
	plt.legend()
	plt.show()

def test_num_nodes():
	X, y = datasets.make_moons(400, noise=0.2)
	num_examples = len(X) # training set size
	nn_input_dim = 2 # input layer dimensionality
	nn_output_dim = 2 # output layer dimensionality 
	learning_rate = 0.01 # learning rate for gradient descent
	reg_lambda = 0.01 # regularization strength
	node_vals = [4,8,16,32,64,128]
	losses_store = []
	for val in node_vals:
		model = build_model(X,val,2)
		model, losses = train(model,X, y, reg_lambda=reg_lambda, learning_rate=learning_rate)
		losses_store.append(losses)
		print losses
	x = np.linspace(0,145,30)
	for i in range(len(losses_store)):
		lab = 'n_nodes = ' + str(node_vals[i])
		plt.plot(x,losses_store[i],label=lab)
	plt.legend()
	plt.show()

print "number of observations:"
num_observations()
print 'noise:'
noise()
print 'regularization:'
reg()
print 'learning rate:'
lr()
print 'hidden nodes:'
test_num_nodes()