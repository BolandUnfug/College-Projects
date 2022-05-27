'''
classification.py

Implements K Nearest Neighbors (KNN) and Naive Bayes classification.

Execution:		>> python3 classification.py [path_to_dataset, str] [class_column_header, str] K=[number_neighbors, int] "X=[input_feature_name, str]" "X=[input_feature_name, str]"
	Examples:		>> python3 classification.py ../data/iris.csv species
					>> python3 classification.py ../data/iris.csv species K=10
					>> python3 classification.py ../data/iris.csv species K=5  "X=petal length (cm)"
					>> python3 classification.py ../data/iris.csv species K=3  "X=petal length (cm)" "X=petal width (cm)"
					>> python3 classification.py ../data/iris.csv species K=3  "X=petal length (cm)" "X=petal width (cm)" "X=sepal length (cm)"

Requires visualization.py, knn.py, and naive bayes.py in the same folder

@author Caitrin Eaton
@date 02/23/2022
'''

import sys							# command line parameters
import os							# file path formatting

import numpy as np					# matrix math
import matplotlib.pyplot as plt		# plotting
from matplotlib import cm			# colormap definitions, e.g. "viridis"

import visualization as vis			# file I/O
import statistics as stats
# import naive_bayes as nb			# naive bayes classification
import pca as pca			# PCA analysis
# import knn							# K nearest neighbors classification

def partition( X, C, pct_train=.75 ):
	'''
	Partition all available samples into separate training and test sets in order to prevent overfitting. \n

	ARGS \n
		X -- (n,m) ndarray, with a row for each datum and a column for each input feature \n
		C -- (n,1) ndarray, with a row for each datum's known target output (class label) \n
		pct_train -- float, the percent of all samples to include in the training set. The rest will be used to test. \n

	RETURN \n
		X_train -- (n_train,m) ndarray, with a row for each datum in the training set and a column for each input feature \n
		C_train -- (n_train,1) ndarray, containing the known output for each input (row) in X_train \n
		X_test -- (n_test,m) ndarray, with a row for each datum in the test set and a column for each input feature \n
		C_test -- (n_test,1) ndarray, containing the known output for each input (row) in X_test
	'''
	n          = X.shape[0]				# number of samples in complete dataset
	print(n)
	k          = X.shape[1]				# number of input features
	print(k)
	n_train    = int(n*pct_train)		# number of samples in training set partition
	n_test     = n - n_train			# number of samples in test set partition
	reorder    = np.arange( 0, n )#.reshape( (n,1) )
	np.random.shuffle( reorder )
	train_rows = reorder[ 0 : n_train ]
	test_rows  = reorder[ n_train : n ]

	# Training set partition
	X_train = X[ train_rows, : ].reshape( (n_train, k) )
	C_train = C[ train_rows, : ].reshape( (n_train, 1) )
	
	# Test set partition
	X_test = X[ test_rows, : ].reshape( (n_test, k) )
	C_test = C[ test_rows, : ].reshape( (n_test, 1) )
	print(X_train.shape)
	print(C_train.shape)
	print(X_test.shape)
	print(C_test.shape)
	return X_train, C_train, X_test, C_test


def evaluate( X, C, C_pred, class_names, title="" ):
	''' Evaluate classifier's ability to predict target class labels Y for the samples X.  \n

	ARGS \n
	-- X: the NxM array of N training samples in M-dimensional space \n
	-- C: the Nx1 array of N known class labels, one per sample in X \n
	-- C_pred: The Nx1 array of N predicted class labels, one per sample in X \n
	-- class_names: list of all class names that exist in the dataset \n
	-- title: Optional title for the confusion matrix \n

	RETURNS \n
	-- Confusion: the KxK confusion matrix, where K is the number of unique class labels in C \n
	-- accuracy: float, 
	'''

	# Count the number of correct and incorrect predictions within each class
	n_classes = len(class_names)
	Confusion = np.zeros((n_classes, n_classes)).astype(int)
	for c in range(n_classes):
		label_true = class_names[c]
		true_c = np.where( C==label_true, 1, 0 ).flatten()
		for p in range(n_classes):
			label_pred = class_names[p]
			pred_c = np.where( C_pred==label_pred, 1, 0 ).flatten()
			Confusion[c,p] = int( np.sum( true_c*pred_c ) )

	# Measure overall accuracy: number correct / total samples
	n = X.shape[0]
	tp = Confusion.diagonal()
	fn = np.sum( Confusion, axis=1 ) - tp
	fp = np.sum( Confusion, axis=0 ) - tp 
	tn = np.sum( Confusion ) - (tp + fn + fp )
	accuracy = np.sum( tp ) / n
	tp_rate   = tp / (tp + fn)
	fp_rate   = fp / (fp + tn)
	precision = tp / (tp + fp)
	
	# Visualize the confusion matrix as a heatmap
	fig = vis.heatmap( Confusion, class_names, title=title )
	ax = plt.gca() 
	ax.set_yticks( np.arange(n_classes) )
	ax.set_yticklabels( class_names )
	ax.set_xlabel( "predicted class")
	ax.set_ylabel( "actual class" )	
	
	# Display results in the terminal
	print(title)
	print(Confusion)
	print(f"Accuracy:  {accuracy:.3f}")
	print(f"TP Rate:   {tp_rate}")
	print(f"FP Rate:   {fp_rate}")
	print(f"Precision: {precision}")

	return Confusion, accuracy, tp_rate, fp_rate, precision



def scatter_2d(X_train, C_train, C_train_pred, X_test, C_test, C_test_pred, acc_test, headers, class_names, title="", ax=None ):
	'''
	Scatterplot the training and test sets, projected onto the first 1-2 dimensions of X.
	
	ARGS: \n
	X_train -- (n_train,m) ndarray, with a row for each datum in the training set and a column for each input feature \n
	C_train -- (n_train,1) ndarray, containing the known output for each input (row) in X_train \n
	C_train_pred -- (n_train,1) ndarray, containing the predicted output for each input (row) in X_train \n
	X_test -- (n_test,m) ndarray, with a row for each datum in the test set and a column for each input feature \n
	C_test -- (n_test,1) ndarray, containing the known output for each input (row) in X_test \n
	C_test_pred -- (n_test,1) ndarray, containing the predicted output for each input (row) in X_test \n
	acc_test -- float, accuracy of the test set \n
	headers -- list of str, the names of the features (columns) in X_train and X_test \n
	class_names -- (n_classes, 1) ndarray, containing all the unique class labels in the entire dataset \n
	title -- str, name of the dataset \n

	RETURN:
	ax (axis reference), the axis on which the new plot appears
	'''

	# Incorporate metadata into axis labels and title
	if ax == None:
		fig, ax = plt.subplots()
	plt.suptitle( title + f"\n Accuracy = {acc_test:.3f}" ) 
	col_x = 0
	col_y = 1
	ax.set_xlabel( headers[ col_x ] )
	ax.set_ylabel( headers[ col_y ] )
	ax.grid( True )
	
	# Visualize training set as circles
	num_classes = len( class_names )
	cmap = cm.get_cmap( "viridis", num_classes ) 
	n_train = X_train.shape[0]
	x = X_train[ :, col_x ].flatten()
	y = X_train[ :, col_y ].flatten()
	C_i = np.argmin( np.abs(class_names*np.ones((n_train,num_classes)) - C_train.reshape((n_train,1))), axis=1 )
	color_true = cmap( C_i/(num_classes-1) )
	ax.scatter( x, y, marker='o', s=35, c=color_true, edgecolor='k', alpha=0.50, label="train" )
	incorrect = C_train.flatten() != C_train_pred.flatten()
	ax.plot( x[incorrect], y[incorrect], 'xr', markersize=10, linewidth=2, label="incorrect (training)" )
	
	# Visualize test set as diamonds
	n_test = X_test.shape[0]
	x = X_test[ :, col_x ].flatten()
	y = X_test[ :, col_y ].flatten()
	C_i = np.argmin( np.abs(class_names*np.ones((n_test,num_classes)) - C_test_pred.reshape((n_test,1))), axis=1 )
	color_pred = cmap( C_i/(num_classes-1) )
	ax.scatter( x, y, marker='d', s=55, c=color_pred, edgecolor='k', alpha=1.0, label="test" )
	incorrect = C_test.flatten() != C_test_pred.flatten()
	ax.plot( x[incorrect], y[incorrect], '+r', markersize=12, linewidth=2, label="incorrect (test)" )
	ax.legend()

	return ax



def scatter_1d(X_train, C_train, C_train_pred, X_test, C_test, C_test_pred, acc_test, headers, class_names, title="", ax=None ):
	'''
	Scatterplot the training and test sets, projected onto the first 1-2 dimensions of X.
	
	ARGS: \n
	X_train -- (n_train,m) ndarray, with a row for each datum in the training set and a column for each input feature \n
	C_train -- (n_train,1) ndarray, containing the known output for each input (row) in X_train \n
	C_train_pred -- (n_train,1) ndarray, containing the predicted output for each input (row) in X_train \n
	X_test -- (n_test,m) ndarray, with a row for each datum in the test set and a column for each input feature \n
	C_test -- (n_test,1) ndarray, containing the known output for each input (row) in X_test \n
	C_test_pred -- (n_test,1) ndarray, containing the predicted output for each input (row) in X_test \n
	acc_test -- float, accuracy of the test set \n
	headers -- list of str, the names of the features (columns) in X_train and X_test \n
	class_names -- (n_classes, 1) ndarray, containing all the unique class labels in the entire dataset \n
	title -- str, name of the dataset \n

	RETURN:
	ax (axis reference), the axis on which the new plot appears
	'''

	# Incorporate metadata into axis labels and title
	if ax == None:
		fig, ax = plt.subplots()
	plt.suptitle( title + f"\n Accuracy = {acc_test:.3f}" ) 
	col_x = 0
	ax.set_xlabel( headers[ col_x ] )
	ax.grid( True )
	
	# Visualize training set as circles
	num_classes = len( class_names )
	cmap = cm.get_cmap( "viridis", num_classes ) 
	x = X_train[ :, col_x ].flatten()
	C_i = np.argmin( class_names - C_train )
	color_true = cmap( C_i /(num_classes-1) )
	ax.scatter( x, np.zeros(x.shape), marker='o', s=35, c=color_true, edgecolor='k', alpha=0.50, label="train" )
	incorrect = C_train.flatten() != C_train_pred.flatten()
	ax.plot( x[incorrect], np.zeros(x[incorrect].shape), 'xr', markersize=10, linewidth=2, label="incorrect (training)" )
	
	# Visualize test set as diamonds
	x = X_test[ :, col_x ].flatten()
	C_i = np.argmin( class_names - C_test_pred )
	color_pred = cmap( C_i /(num_classes-1) )
	ax.scatter( x, np.zeros(x.shape), marker='d', s=55, c=color_pred, edgecolor='k', alpha=1.0, label="test" )
	incorrect = C_test.flatten() != C_test_pred.flatten()
	ax.plot( x[incorrect], np.zeros(x[incorrect].shape), '+r', markersize=12, linewidth=2, label="incorrect (test)" )
	ax.legend()

	return ax


def predict_knn(X, Cx, S, k=1 ):
	'''Use K Nearest Neighbors to classify new samples S, given training set X with known classes Cx. \n
	
	ARGS \n
	X -- (n, m) ndarray of n training samples, each m input features wide \n
	Cx -- (n, 1) ndarray of n training samples' known class labels \n
	S -- (p, m) ndarray of p unlabeled samples, each m input features wide \n
	k -- int, number of nearest neighbors in the training set to use when predicting a new sample's class label, \n
	verbose -- bool, results are printed to the terminal only if verbose is True
	'''
	
	# Keep numpy from throwing an interdimensional tantrum when S contains only 1 sample
	m = X.shape[1]
	if len(S.shape)<2:
		S.reshape( (1, m) )
	
	# Classify the sample(s) in S
	Cs = np.zeros( (S.shape[0],1) ) - 1
	for si in range(S.shape[0]):
		s = S[si,:]#.reshape((1,m))
		distances = np.sum((X-s)**2,axis=1)	# Compute distances (does not have to be Euclidean)
		idx = np.argsort(distances,axis=0)			# Sort neighbors, nearest to farthest
		Cx_sort = Cx[ idx ]							# Sort classes, to match
		class_names, class_counts = np.unique( Cx_sort[0:k], return_counts=True )
		Cs[si] = class_names[ np.argmax( class_counts ) ]
	
	return Cs


def classify_knn( X_train, C_train, X_test, C_test, headers, k=1, title="" ):
	'''Train and test a KNN classifier, given labeled training and test sets. \n

	ARGS \n
	X_train -- (n_train,m) ndarray, with a row for each datum in the training set and a column for each input feature \n
	C_train -- (n_train,1) ndarray, containing the known output for each input (row) in X_train \n
	X_test -- (n_test,m) ndarray, with a row for each datum in the test set and a column for each input feature \n
	C_test -- (n_test,1) ndarray, containing the known output for each input (row) in X_test \n
	headers -- list of str, the names of the features (columns) in X_train and X_test \n
	k -- int, number of nearest neighbors in the training set to use when predicting a new sample's class label \n
	title -- str, name of the dataset \n
	verbose -- bool, results are printed to the terminal only if verbose is True

	RETURNS \n
	C_test_pred -- (n_test,1) ndarray, containing the predicted output for each input (row) in X_test \n
	Conf_test -- (num_classes, num_classes) ndarray, containing the confusion matrix for the test set \n
	acc_test -- float, accuracy of the classifier in the test set \n
	Conf_train -- (num_classes, num_classes) ndarray, containing the confusion matrix for the test set \n
	acc_train -- float, accuracy of the classifier in the training set
	'''

	m = X_train.shape[1]
	n_train = X_train.shape[0]
	n_test = X_test.shape[0]
	X = np.vstack((X_train, X_test))
	C = np.vstack((C_train.reshape((n_train,1)), C_test.reshape((n_test,1))))	
	class_names = np.unique( C_train )
	num_classes = len(class_names)

	# Evaluate performance of the classifier on training set
	print( f"\n\n{title} : KNN CLASSIFIER" )
	print( f"{num_classes:d} classes in {m:d} dimensional feature space, with K={k:d} nearest neighbors" )
	print("\nTRAINING SET PERFORMANCE:")
	C_train_pred = predict_knn( X_train, C_train, X_train, k )
	Conf_train, acc_train, _, _, _ = evaluate( X_train, C_train, C_train_pred, class_names, title + f": KNN Training Set, K={k:d}")
	
	# Evaluate performance of the classifier on test set
	print("\n\nTEST SET PERFORMANCE:")
	C_test_pred = predict_knn( X_train, C_train, X_test, k )
	Conf_test, acc_test, _, _, _ = evaluate( X_test, C_test, C_test_pred, class_names, title + f": KNN Test Set, K={k:d}")

	# Visualize the classifier's output as a scatterplot projected onto the first 2 axes of X
	if X.shape[1] > 1:
		fig = scatter_2d(X_train, C_train, C_train_pred, X_test, C_test, C_test_pred, acc_test, headers, class_names, title=title+f": KNN, K={k:d}" )
	else:
		fig = scatter_1d(X_train, C_train, C_train_pred, X_test, C_test, C_test_pred, acc_test, headers, class_names, title=title+f": KNN, K={k:d}" )
	
	return C_test_pred, Conf_test, acc_test, Conf_train, acc_train 
	

def train_nb(X, C):
	'''Train a Naive Bayes classifier by determining the means and standard deviations of each class in the training set X. \n
	
	ARGS \n
	X -- (n, m) ndarray of n training samples, each m input features wide \n
	C -- (n, 1) ndarray of n training samples' known class labels \n

	RETURNS \n
	means -- (k, m) ndarray of k means (one per class), each m input features wide \n
	variances -- (k, m) ndarray of k variances (one per class), each m input features wide \n
	priors -- (k,) ndarray of k prior probabilities (one per class)
	'''				
	
	# How many classes (K) are there, and how many samples in each?
	n = X.shape[0]
	m = X.shape[1]
	class_names, class_counts = np.unique(C, return_counts=True)
	num_classes = len(class_names)

	# Prior probability of a sample belonging to each class
	priors = class_counts / n
	
	# Compute the mean and standard deviation of each class
	means = np.zeros((num_classes, m))
	variances  = np.zeros((num_classes, m))
	for c in range( num_classes ):
		label = class_names[ c ]
		has_label = (C == label).reshape((n,))
		Xc = X[ has_label, : ]
		means[ c, : ] = np.mean( Xc, axis=0 )
		variances[ c, : ] = np.var( Xc, axis=0 )

	return means, variances, priors


def predict_nb(means, variances, priors, X, class_names):
	'''Use a trained Naive Bayes classifier to predict the class labels of new samples X. \n
	
	ARGS \n
	means -- (k, m) ndarray of the mean of each class, each m features wide \n
	variances -- (k, m) ndarray of the variance of each class, each m features wide \n
	priors -- (k,) ndarray of the prior probability a sample belonging to each class \n
	X -- (n,m) ndarray of n samples to be classified, each m features wide \n
	class_names -- (k,) ndarray of the label of each class

	RETURNS \n
	C_pred -- (n,1) ndarray of predicted class labels for each sample in X \n
	P_pred -- (n,1) ndarray of our confidence that each sample belongs to its predicted class
	'''				
	
	# Class descriptors
	num_classes = means.shape[0]
	
	# Calculate the probability each sample belongs to each class
	n = X.shape[0]
	P_c_given_x = np.zeros( (n,num_classes) )	
	for c in range(num_classes):
		height = 1 / (np.sqrt(2 * np.pi * variances[c,:]) )
		P_x_given_c = np.prod( height * np.exp( -(X-means[c,:])**2 / (2*variances[c,:]) ), axis=1 )
		P_c_given_x[:,c] = P_x_given_c * priors[c]
		
	# Assign each sample to the class with the greatest probability
	C_i = np.argmax( P_c_given_x, axis=1 )	# axis=1 because we want the max in each ROW
	C_pred = class_names[ C_i ]
	P_pred = P_c_given_x[ :, C_i ] / np.sum( P_c_given_x, axis=1 )
	
	return C_pred, P_pred


def classify_nb( X_train, C_train, X_test, C_test, headers, title="" ):
	'''Train and test a Naive Bayes classifier, given labeled training and test sets. \n

	ARGS \n
	X_train -- (n_train,m) ndarray, with a row for each datum in the training set and a column for each input feature \n
	C_train -- (n_train,1) ndarray, containing the known output for each input (row) in X_train \n
	X_test -- (n_test,m) ndarray, with a row for each datum in the test set and a column for each input feature \n
	C_test -- (n_test,1) ndarray, containing the known output for each input (row) in X_test \n
	headers -- list of str, the names of the features (columns) in X_train and X_test \n
	title -- str, name of the dataset \n
	verbose -- bool, results are printed to the terminal only if verbose is True

	RETURNS \n
	C_test_pred -- (n_test,1) ndarray, containing the predicted output for each input (row) in X_test \n
	Conf_test -- (num_classes, num_classes) ndarray, containing the confusion matrix for the test set \n
	acc_test -- float, accuracy of the classifier in the test set \n
	Conf_train -- (num_classes, num_classes) ndarray, containing the confusion matrix for the test set \n
	acc_train -- float, accuracy of the classifier in the training set
	'''

	m = X_train.shape[1]
	n_train = X_train.shape[0]
	n_test = X_test.shape[0]
	X = np.vstack((X_train, X_test))
	print(C_train.shape)
	print(C_test.shape)
	C = np.vstack((C_train.reshape((n_train,1)), C_test.reshape((n_test,1))))	
	class_names = np.unique( C )
	num_classes = len(class_names)

	# Train the Naive Bayes classifier
	means, variances, priors = train_nb( X_train, C_train )

	# Print trained Naive Bayes model's characteristics to the terminal
	print( f"\n\n{title} : NAIVE BAYES CLASSIFIER" )
	print( f"{num_classes:d} classes in {m:d} dimensional feature space" )
	for c in range(means.shape[0]):
		# Display class sats
		print("\tclass", c )
		print("\t\tmean :    ", means[c,:])
		print("\t\tvariance :", variances[c,:])
		print("\t\tpriors :  ", priors[c])
	

	# Evaluate performance of the classifier on training set
	print("\nTRAINING SET PERFORMANCE:")
	C_train_pred, _ = predict_nb( means, variances, priors, X_train, class_names )
	Conf_train, acc_train, _, _, _ = evaluate( X_train, C_train, C_train_pred, class_names, title+": NB Training Set")
	
	# Evaluate performance of the classifier on test set
	print("\n\nTEST SET PERFORMANCE:")
	C_test_pred, _ = predict_nb( means, variances, priors, X_test, class_names )
	Conf_test, acc_test, _, _, _ = evaluate( X_test, C_test, C_test_pred, class_names, title+": NB Test Set")

	# Visualize the classifier's output as a scatterplot projected onto the first 2 axes of X
	fig, ax = plt.subplots()
	cmap = cm.get_cmap( "viridis", num_classes ) 
	
	for c in range(num_classes):
		label = class_names[c]
		color = cmap( c/(num_classes-1) )
		if X.shape[1] > 1:
			cov = np.eye(m) * variances[c,:]
			for n_std in range(1,4):
				stats.gaussian_ellipse( means[c,:], cov, ax=ax, n_std=n_std, color=color )
		else:
			x_line = np.linspace( np.min(X, axis=0), np.max(X, axis=0), 100 ).reshape((100,1))
			y_line = 1 / (np.sqrt(2*np.pi*variances[c])) * np.exp( -(x_line - means[c])**2 / (2*variances[c]) )
			y_line = y_line.reshape((100,1))
			ax.plot( x_line, y_line, '-', color=color, linewidth=2, alpha=1.0, label=class_names[c] )
	if X.shape[1] > 1:
		scatter_2d(X_train, C_train, C_train_pred, X_test, C_test, C_test_pred, acc_test, headers, class_names, title=title+": Naive Bayes", ax=ax )
	else:
		scatter_1d(X_train, C_train, C_train_pred, X_test, C_test, C_test_pred, acc_test, headers, class_names, title=title+": Naive Bayes", ax=ax )

		
	# Visualize the trained model's means and variances
	fig_mean = vis.heatmap( means, headers, title=title + ": Naive Bayes Means" )
	ax = plt.gca() 
	ax.set_yticks( np.arange(num_classes) )
	ax.set_yticklabels( class_names )
	ax.set_xlabel( "features")
	ax.set_ylabel( "classes" )	
	fig_var = vis.heatmap( variances, headers, title=title + ": Naive Bayes Variances" )
	ax = plt.gca() 
	ax.set_yticks( np.arange(num_classes) )
	ax.set_yticklabels( class_names )
	ax.set_xlabel( "features")
	ax.set_ylabel( "classes" )	

	return C_test_pred, Conf_test, acc_test, Conf_train, acc_train 
	

def main( argv ):
	''' Parse command line arguments: 
		-- argv[0] is always the name of the program run from the terminal
		-- argv[1] should be the path of a data file (e.g. *.DATA or *.CSV) 
		-- argv[2] should be the name of the target feature that the user wants to predict (i.e. the Y-axis header)
	'''

	# Since we don't care too much about decimal places, today, let's make the
	# output a little more human friendly. Heads up: This isn't always a good idea!
	np.set_printoptions(precision=3, suppress=True)

	# Determne the input file's path: either it was supplied in the commandline, or we should use iris as a default
	if len(argv) > 1:
		filepath = argv[1].strip()
	else:
		#filepath = "../data/iris_preproc.csv"
		current_directory = os.path.dirname(__file__)
		filepath = os.path.join(current_directory, "..", "data", "iris_preproc.csv")

	# Read in the dataset and remove NaNs
	
	data, headers, title = vis.read_csv( filepath )
	print(data.shape)
	data, headers = vis.remove_nans( data, headers )
	print(data.shape)
	# Let the user name a class label feature (the target to predict), input feature(s), and/or the number of nearest
	# neighbors, k. Use commandline parameter prefixes: "K=" for number of nearest neighbors, "C=" for class feature, 
	# "X=" for each input feature. By default it will set K=1 and use the rightmost column as the class label feature.
	k = 1
	
	n = data.shape[0]
	C_idx = -1
	C_header = headers[C_idx]
	C = data[:,C_idx].reshape((n,1))
	X_headers = []
	X = None
	if len(argv) > 2:
		for param in argv[2:]:

			if "C=" in param:
				# Find the column that contains the user's chosen feature name 
				C_param = param.split("=")[1]
				if C_param in headers:
					C_idx = headers.index( C_param )	# CAUTION: this will crash if the user's input does not match any element in the list of headers
					C_header = headers[ C_idx ]
					C = data[ :, C_idx ].reshape((n,1))
					print( f"Target class feature selected: {C_header}" )
				else:
					print( f"\nWARNING: '{C_param}' not found in the headers list: {headers}. No target class feature selected.\n" )

			elif "K=" in param:	
				# Let the user specify the maximum polynomial degree that they want to use in the model with the syntax "D=2", where 2 could be any integer
				k_param = param.split("=")[1]
				k = int( k_param )
				print( f"KNN configured to use {k} nearest neighbors" )

			elif "X=" in param:
				# Let the user specify 1 or more input features (X axes), with 1 or more "X=feature_name" commandline parameters
				X_param = param.split("=")[1]
				if X_param in headers:
					X_idx = headers.index( X_param )	# CAUTION: this will crash if the user's input does not match any element in the list of headers
					X_headers.append( headers[ X_idx ] )
					if len(X_headers) > 1:
						X = np.hstack( (X, data[ :, X_idx ].reshape((n,1))) )
					else: 
						X = data[ :, X_idx ].reshape((n,1))
					print( f"Input X feature(s) selected: {X_headers}" )
				else:
					print( f"\nWARNING: '{X_param}' not found in the headers list: {headers}. No input X feauture selected.\n" )	

	# If the user has not specified any X features, use every feature except C as input, 1 at a time
	if len(X_headers)==0:
		X = np.delete( data, C_idx, axis=1 )
		X_headers = headers[:]
		X_headers.pop( C_idx ) 

	# If the user has specified both X and C features, we can ignore the rest of the dataset
	elif len(C_header) > 0 and len(X_headers) > 0:
		data = np.hstack((X,C))
		headers = X_headers + [ C_header ]

	# Train both a KNN classifier and a NB classifier with the same training & test sets, for comparison.
	X_train, C_train, X_test, C_test = partition( X, C, 0.5 )
	classify_nb( X_train, C_train, X_test, C_test, X_headers, title=f"{title} {C_header}" )
	classify_knn( X_train, C_train, X_test, C_test, X_headers, k=k, title=f"{title} {C_header}" )
	
	# Z-score normalization
	title2 = title + " Z-Transform"
	X_norm_train = (X_train - X_train.mean( axis=0 )) / X_train.std( axis=0 )
	X_norm_test = (X_test - X_test.mean( axis=0 )) / X_test.std( axis=0 )
	classify_nb( X_norm_train, C_train, X_norm_test, C_test, X_headers, title=f"{title2} {C_header}" )
	classify_knn( X_norm_train, C_train, X_norm_test, C_test, X_headers, k=k, title=f"{title2} {C_header}" )
	
	# PCA thing
	title3 =  "PCA " + title # getting C_test and C_train mixed up,
	X_train_norm, X_train_mean, X_train_std = pca.z_transform(X_train)
	X_test_norm = (X_test - X_train_mean) / X_train_std
	Y_train, P, e_scaled = pca.pca_cov(X_train_norm)
	Y_test = X_test_norm @ P
	Y_train = Y_train[:,0:2]
	Y_test = Y_test[:,0:2]

	classify_nb( Y_train, C_train, Y_test, C_test,headers=["P0","P1"], title=f"{title3} {C_header}" )
	classify_knn( Y_train, C_train, Y_test, C_test,headers=["P0","P1"], k=k, title=f"{title3} {C_header}" )
    
if __name__=="__main__":
	main( sys.argv )
	plt.show()