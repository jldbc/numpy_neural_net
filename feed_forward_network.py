import numpy as np
import math

def activation_function(x):
	return np.tanh(x)

def softmax(x):
	pass

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def loss(yhat, y, measure):
	if measure == "mse":
		mse = np.square(y-yhat).mean()
		return mse
	if measure == "cross_entropy":
		print "oops, haven't implemented this yet"


def feed_forward(X, y, nodes=[16,8], n_iter=1000, model_type="regression"):
	nodes.append(1) #single node for final output
 	n_obs = X.shape[0]
	depth = len(nodes)
	#w needs to know dim of previous layer for initialization
	prev_dim = [X.shape[1]]
	for i in range(depth-1):
		prev_dim.append(nodes[i])
	#weights: list of weight matrices. (n_nodes for prev. layer) x (n_nodes for next layer).
	weights = [np.random.rand(1,nodes[i]) for i in range(depth)]
	weights = [np.repeat(weights[i],prev_dim[i],axis=0) for i in range(depth)]
	#bias: n_layers x 1 array. A scalar bias value for each layer.
	bias = np.asarray([np.random.rand(1) for i in range(depth)])
	#train the network
	for i in range(n_iter):
		h = X #initial nodes are the column values for each observation
		for k in range(depth):
			if k != depth:
				a = h.dot(weights[k]) + bias[k] #(layer input * weights) + bias node = activation input
				h = activation_function(a) #activation(Wx + b) = next layer's input
			elif k == depth: #different activation for the output layer
				a = h.dot(weights[k]) + bias[k] #no additional function if regression. softmax if classification.
				if model_type == "classification":
					h = softmax(a)
				elif model_type == "regression":
					h = a
	iter_loss = loss(h, y, measure="mse")
	print iter_loss
	return h

def backprop():
	pass


def main():
	X = np.linspace(0,10,100)
	y = [np.sin(i) for i in X]
	X = np.asmatrix(X).T
	y = np.asmatrix(y).T
	feed_forward(X,y,nodes=[128,64, 32, 16])



#