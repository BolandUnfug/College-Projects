'''	perceptron.py

	Implement a single perceptron (artificial neuron), and train it to solve a logical AND operation.

	Execution: python3 perceptron.py

TODO: 
@author Boland Unfug
@date 5/5/2022
'''

import numpy as np
import matplotlib.pyplot as plt
from activation_functions import *

	
# TODO: Complete this function to train a perceptron
def train_perceptron( X, Y, learning_rate=0.5, threshold=0, activation_function=step_activation ):
	''' Train a perceptron to predict the training targets Y given training inputs X. 

	ARGS:
		X: (n,m) ndarray of training inputs, in which each row represents 1 sample and each column represents 1 feature
		Y: (n,1) ndarray of training targets
		learning_rate: float, determines how strongly residuals impact weights during each iteration of training.

	RETURNS:
		W: (m+1,) ndarray of weights for each column of X + the bias term (intercept) 
		Y_pred: (n,1) ndarray of predictions for each sample (row) of X
		mse: float, mean squared error
	'''

	# Create a figure window for tracking mse over time
	fig = plt.figure()
	plt.title( "Perceptron training" )
	plt.xlabel( "epoch" )
	plt.ylabel( "MSE" )
	plt.grid( True )

	# TODO: Horizontally stack a columne of ones (for the bias term) onto the right side of X
	n, m = X.shape
	A = np.hstack((X, np.ones((n,1))))

	# TODO: Initialize weights (including the bias) to small random numbers
	W = np.random.rand(m+1,1) *2 -1

	# TODO: Loop over training set until error is acceptably small, or iteration cap is reached	
	epoch = 0
	max_epochs = 100
	min_error = .0001
	mse = min_error *2
	Y_pred = None
	while ((epoch < max_epochs) and (min_error < mse)):
		for i in range(n):
			sample = A[i,:].reshape((1,m+1))# grab a sample
			pred = activation_function(sample,W,threshold)
			residual = pred - Y[i]
			for dim in range(m+1):
				# bias and weights get updated
				W[dim,0] = W[dim,0] - sample[0,dim] * residual * learning_rate

		Y_pred = activation_function(A,W,threshold)
		square_error = (Y_pred - Y)**2
		mse = np.mean(square_error)
		epoch += 1
		plt.plot(epoch, mse, 'ko')
		plt.pause(0.001)
	return W, Y_pred, mse


def test_logical_and():
	''' Train a perceptron to perform a logical AND operation. '''
	print( "\nTESTING LOGICAL AND" )
	truth_table = np.array( [[0,0,0], [0,1,0], [1,0,0], [1,1,1]] )
	n = truth_table.shape[0]
	X = truth_table[:, 0:2]
	Y = truth_table[:, 2].reshape((n,1))
	learning_rate = 0.5
	W, Y_pred, mse = train_perceptron( X, Y, learning_rate, activation_function=step_activation )

	# Display results in the temrinal
	print( "\nWeights:", W.T )
	print( "MSE:", mse )
	print( "------------------" )
	print( " X0  X1 Y  Y* " )
	print( "------------------" )
	print( np.hstack((truth_table, Y_pred)) )
	print( "------------------" )
	
	plt.title( "AND" )
	plt.show()


def test_logical_or():
	''' Train a perceptron to perform a logical OR operation. '''
	print( "\nTESTING LOGICAL OR" )
	truth_table = np.array( [[0,0,0], [0,1,1], [1,0,1], [1,1,1]] )
	n = truth_table.shape[0]
	X = truth_table[:, 0:2]
	Y = truth_table[:, 2].reshape((n,1))
	learning_rate = 0.5 #0.005 is good for ReLU, softplus
	W, Y_pred, mse = train_perceptron( X, Y, learning_rate, activation_function=step_activation )
	
	# Display results in the temrinal
	print( "\nWeights:", W.T )
	print( "MSE:", mse )
	print( "------------------" )
	print( " X0  X1 Y  Y* " )
	print( "------------------" )
	print( np.hstack((truth_table, Y_pred)) )
	print( "------------------" )

	plt.title( "OR" )
	plt.show()


def test_logical_xor():
	''' Train a perceptron to perform a logical XOR operation. '''
	print( "\nTESTING LOGICAL XOR" )
	truth_table = np.array( [[0,0,0], [0,1,1], [1,0,1], [1,1,0]] )
	n = truth_table.shape[0]
	X = truth_table[:, 0:2]
	Y = truth_table[:, 2].reshape((n,1))
	learning_rate = 0.5
	W, Y_pred, mse = train_perceptron( X, Y, learning_rate, activation_function=step_activation )
	
	# Display results in the temrinal
	print( "\nWeights:", W.T )
	print( "MSE:", mse )
	print( "------------------" )
	print( " X0  X1 Y  Y* " )
	print( "------------------" )
	print( np.hstack((truth_table, Y_pred)) )
	print( "------------------" )

	plt.title( "XOR" )
	plt.show()


def test_line():
	''' Train a perceptron to recreate a straight line. '''
	print( "\nTESTING STRAIGHT LINE" )
	n = 50
	X = np.linspace( -10, 10, n ).reshape((n,1))
	m = (np.random.random() - 0.5) * 20
	b = (np.random.random() - 0.5) * 20
	Y = m*X + b
	learning_rate = 0.05
	W, Y_pred, mse = train_perceptron( X, Y, learning_rate, activation_function=linear_activation )
	plt.title( "LINE TEST" )
	
	# Display results in the temrinal
	print( "\nWeights:", W.T )
	print( "MSE:", mse )
	print( "-------------------------" )
	print( " X        Y       Y* " )
	print( "-------------------------" )
	print( np.hstack((X, Y, Y_pred)) )
	print( "-------------------------" )

	plt.figure()
	plt.plot( X, Y, 'ob', alpha=0.5, label="Y" )
	plt.plot( X, Y_pred, 'xk', label="Y*" )
	plt.grid( True )
	plt.xlabel( "X" )
	plt.ylabel( "Y" )
	plt.title( f"Target:  Y  = {m}*X + {b}\nLearned: Y* = {W[0]}*X + {W[1]}" )
	plt.show()


if __name__=="__main__":
	test_logical_and()
	test_logical_or()
	test_logical_xor()
	test_line()