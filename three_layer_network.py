import numpy as np 
import math
from sklearn import datasets

def relu(X):
	return np.maximum(X, 0)

def relu_derivative(X):
	return 1. * (X > 0)

def build_model(X,hidden_nodes,output_dim=2):
    model = {}
    input_dim = X.shape[1]
    model['W1'] = np.random.randn(input_dim, hidden_nodes) / np.sqrt(input_dim)
    model['b1'] = np.zeros((1, hidden_nodes))
    model['W2'] = np.random.randn(hidden_nodes, output_dim) / np.sqrt(hidden_nodes)
    model['b2'] = np.zeros((1, output_dim))
    return model

def feed_forward(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = x.dot(W1) + b1
    #a1 = np.tanh(z1)
    a1 = relu(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    out = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return z1, a1, z2, out

def calculate_loss(model,X,y,reg_lambda):
    num_examples = X.shape[0]
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation to calculate our predictions
    z1, a1, z2, out = feed_forward(model, X)
    probs = out / np.sum(out, axis=1, keepdims=True)
    # Calculating the loss
    corect_logprobs = -np.log(probs[range(num_examples), y])
    loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * loss

def backprop(X,y,model,z1,a1,z2,output,reg_lambda):
    delta3 = output
    delta3[range(X.shape[0]), y] -= 1  #yhat - y
    dW2 = (a1.T).dot(delta3)
    db2 = np.sum(delta3, axis=0, keepdims=True)
    #delta2 = delta3.dot(model['W2'].T) * (1 - np.power(a1, 2)) #if tanh
    delta2 = delta3.dot(model['W2'].T) * relu_derivative(a1) #if ReLU
    dW1 = np.dot(X.T, delta2)
    db1 = np.sum(delta2, axis=0)
    # Add regularization terms
    dW2 += reg_lambda * model['W2']
    dW1 += reg_lambda * model['W1']
    return dW1, dW2, db1, db2


def train(model, X, y, num_passes=10000, reg_lambda = .1, learning_rate=0.1):
    # Batch gradient descent
    done = False
    previous_loss = float('inf')
    i = 0
    losses = []
    while done == False:  #comment out while performance testing
    #while i < 150:
    	#feed forward
        z1,a1,z2,output = feed_forward(model, X)
        #backpropagation
        dW1, dW2, db1, db2 = backprop(X,y,model,z1,a1,z2,output,reg_lambda)
        #update weights and biases
        model['W1'] -= learning_rate * dW1
        model['b1'] -= learning_rate * db1
        model['W2'] -= learning_rate * dW2
        model['b2'] -= learning_rate * db2
        if i % 1000 == 0:
        	loss = calculate_loss(model, X, y, reg_lambda)
        	losses.append(loss)
        	print "Loss after iteration %i: %f" %(i, loss)  #uncomment once testing finished, return mod val to 1000
        	if (previous_loss-loss)/previous_loss < 0.01:
        		done = True
        		#print i
        	previous_loss = loss
        i += 1
    return model, losses

def main():
	#toy dataset
	X, y = datasets.make_moons(16, noise=0.10)
	num_examples = len(X) # training set size
	nn_input_dim = 2 # input layer dimensionality
	nn_output_dim = 2 # output layer dimensionality 
	learning_rate = 0.01 # learning rate for gradient descent
	reg_lambda = 0.01 # regularization strength
	model = build_model(X,20,2)
	model, losses = train(model,X, y, reg_lambda=reg_lambda, learning_rate=learning_rate)
	output = feed_forward(model, X)
	preds = np.argmax(output[3], axis=1)

if __name__ == "__main__":
    main()
