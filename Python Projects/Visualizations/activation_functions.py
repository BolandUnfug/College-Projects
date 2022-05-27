'''	activation_functions.py

	Defines several activation functions that can be used in perceptron.py.

	This file cannot be executed on its own, only imported for use in other files, for example: 
		from activation_function.py import *
	
	Students: Add any additional activation functions that you like!

@author Caitrin Eaton
@date May 2022
'''

import numpy as np

def step_activation( X, W, threshold=0 ):
	''' A step activation function.

	ARGS:
		X: (n,m+1) ndarray of inputs, in which each row represents 1 sample + normal homogeneous coordinate, and each column represents 1 feature
		W: (m+1,1) ndarray of weights for each input feature as well as the bias term (intercept)
		threshold: float, determines how strongly residuals impact weights during each iteration of training.

	RETURNS:
		activation: (n,1) ndarray, the perceptron's output in response to each sample (row) in X
	'''
	n = X.shape[0]
	weighted_sum = X @ W
	activation = np.zeros((n,1))
	activation[ weighted_sum > threshold ] = 1.0
	return activation


def sigmoid_activation( X, W, threshold=0 ):
	''' A logistic sigmoid activation function

	ARGS:
		X: (n,m+1) ndarray of inputs, in which each row represents 1 sample + normal homogeneous coordinate, and each column represents 1 feature
		W: (m+1,1) ndarray of weights for each input feature as well as the bias term (intercept)
		threshold: float, determines how strongly residuals impact weights during each iteration of training.

	RETURNS:
		activation: (n,1) ndarray, the perceptron's output in response to each sample (row) in X
	'''
	weighted_sum = X @ W
	activation = 1 / (1 + np.exp( -(weighted_sum - threshold) ))
	return activation


def relu_activation( X, W, threshold=0 ):
	''' A rectified linear ("ReLU") activation function

	ARGS:
		X: (n,m+1) ndarray of inputs, in which each row represents 1 sample + normal homogeneous coordinate, and each column represents 1 feature
		W: (m+1,1) ndarray of weights for each input feature as well as the bias term (intercept)
		threshold: float, determines how strongly residuals impact weights during each iteration of training.

	RETURNS:
		activation: (n,1) ndarray, the perceptron's output in response to each sample (row) in X
	'''
	activation = X @ W
	activation[ activation < threshold ] = threshold
	return activation

	

def softplus_activation( X, W, threshold=0 ):
	''' A softplus activation function is similar in shape to a smoothed (curvy) ReLU.

	ARGS:
		X: (n,m+1) ndarray of inputs, in which each row represents 1 sample + normal homogeneous coordinate, and each column represents 1 feature
		W: (m+1,1) ndarray of weights for each input feature as well as the bias term (intercept)
		threshold: float, determines how strongly residuals impact weights during each iteration of training.

	RETURNS:
		activation: (n,1) ndarray, the perceptron's output in response to each sample (row) in X
	'''
	weighted_sum = X @ W
	activation = np.log( 1 + np.exp(weighted_sum - threshold) )
	return activation



def linear_activation( X, W, threshold=0 ):
	''' A linear activation function.

	ARGS:
		X: (n,m+1) ndarray of inputs, in which each row represents 1 sample + normal homogeneous coordinate, and each column represents 1 feature
		W: (m+1,1) ndarray of weights for each input feature as well as the bias term (intercept)
		threshold: Not used in a linear activation function.

	RETURNS:
		activation: (n,1) ndarray, the perceptron's output in response to each sample (row) in X
	'''
	activation = X @ W
	return activation