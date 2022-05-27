'''
classification.py

Implements K Nearest Neighbors (KNN) and Naive Bayes classification.

Execution:		>> python3 classification.py [path_to_dataset, str] "C=[class_column_header, str]" K=[number_neighbors, int] "X=[input_feature_name, str]" "X=[input_feature_name, str]"
	Examples:		>> python3 classification.py ../data/iris.csv "C=species"
					>> python3 classification.py ../data/iris.csv "C=species" K=10
					>> python3 classification.py ../data/iris.csv "C=species" K=5  "X=petal length (cm)"
					>> python3 classification.py ../data/iris.csv "C=species" K=3  "X=petal length (cm)" "X=petal width (cm)"

Requires visualization.py and statistics.py in the same folder

MLVT students: To jump to sections of code left for you to fill, CTRL+F for "TODO:"
You will find 4 functions that need your help in order to execute:
-- I.   partition()
-- II.  predict_knn()
-- III. train_nb()
-- IV.  predict_nb()

TODO:
@author Boland Unfug
@date March 2 2022
'''

import sys							# command line parameters
import os							# file path formatting

import numpy as np					# matrix math
import matplotlib.pyplot as plt		# plotting
from matplotlib import cm			# colormap definitions, e.g. "viridis"

import visualization as vis			# file I/O
import regression_poly as reg		# Partitioning
import statistics as stats			# drawing gaussian ellipses

# I MLVT students DO need to complete this function
def partition( X, C, pct_train=0.50 ):
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
	
	X_train, C_train, X_test, C_test = reg.partition(X=X, Y=C,pct_train=pct_train)

	return X_train, C_train, X_test, C_test


# II. MLVT students DO need to complete this function
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
	
	# 2. TODO: Classify the sample(s) in S, by placing your KNN algorithm from class inside of a for loop that loops over the rows of S.
	Cs_pred = np.zeros( (S.shape[0],1) ) - 1	# Initializes all predictions to "-1" (an uncommon class label)
	for si in range(S.shape[0]):
		s = S[si,:]

		# 3. Vectorize the distance calculations between ONE new sample, S[i,:] and ALL training samples in X.
		dist = np.sqrt(np.sum((X-s)**2, axis=1))
		# 4. Determine row indices that would sort these distances from nearest to farthest
		idx = np.argsort (dist, axis=0)
		# 5. Arrange training set's class labels in order from nearest to farthest
		Cx_sorted = Cx[idx]
		# 6. Choose the most common class label among S[si,:]'s K nearest neighbors as this new sample's predicted class label
		class_names, class_counts = np.unique(Cx_sorted[0:k], return_counts=True)
		#print("counts ", class_counts , " names " ,class_names)
		Cs_pred[si,:] = class_names[np.argmax(class_counts)]
	return Cs_pred
	

# III. MLVT students DO need to complete this function
def train_nb(X, C):
	'''Train a Naive Bayes classifier by determining the means and standard deviations of each class in the training set X. \n
	
	ARGS \n
	X -- (n, m) ndarray of n training samples, each m input features wide \n
	C -- (n, 1) ndarray of n training samples' known class labels \n

	RETURNS \n
	means -- (k, m) ndarray of k means (one per class), each m input features wide \n
	variances -- (k, m) ndarray of k variances (one per class), each m input features wide \n
	priors -- (k,1) ndarray of k prior probabilities (one per class)
	'''		
	# 0. TODO: Delete this when you're ready to test this function
	priors	=  []	
	# 1. TODO: How many classes exist, and how many training samples belong to each class?
	# am I supposed to get each unique label? is this just C? currently gets each unique species for iris_preproc
	class_labels, class_counts = np.unique( C, return_counts=True )	# determine if there is more than one class label present
	num_classes = len( class_labels )
	#print(str(class_labels) + " and " + str(class_counts))
	

	# 2. TODO: Compute the prior probability of a sample belonging to each class
	# ONLY WORKS FOR NUMBERED CLASSES
	priors = class_counts / sum(class_counts)

	#print("priors" + str(priors)) # these numbers make sense

	# 3. TODO: Compute the mean and variance of each class
	# can probably be vectorized, but I will do that later.
	# correctly gets the sums, but for the whole set, and not each class
	
	means = np.zeros((num_classes, X.shape[1]))
	variances = np.zeros((num_classes, X.shape[1]))
	for i in range( num_classes ):

		# Find the samples that belong to this specific class
		label = class_labels[ i ]

		#print(label)
		#print(C[:,0])
		#print(cthingie)
		has_lbl = C[:, 0] == label
		#print(has_lbl)
		# TODO:
		# Compute the distribution within this specific class
		# same as other thing, but you specify the row with i? or is it column?
		means[i,:] = np.mean(X[has_lbl , :], axis=0)
		
		variances[i,:] = np.var(X[has_lbl , :], axis=0)
		
		#print(mean)
		#print(var)

	#print("means " + str(means)) # these numbers make sense, but only for the total. need to figure out how to do it for each class.
	#print("variances " + str(variances)) # these numbers make sense, but only for the total. need to figure out how to do it for each class.
	
	return means, variances, priors


# IV. TODO: MLVT students DO need to complete this function
def predict_nb(means, variances, priors, X):
	'''Use a trained Naive Bayes classifier to predict the class labels of new samples X. \n
	
	ARGS \n
	means -- (k, m) ndarray of the mean of each class, each m features wide \n
	variances -- (k, m) ndarray of the variance of each class, each m features wide \n
	priors -- (k,) ndarray of the prior probability a sample belonging to each class \n
	X -- (n,m) ndarray of n samples to be classified, each m features wide \n
	
	RETURNS \n
	C_pred -- (n,1) ndarray of predicted class labels for each sample in X \n
	P_pred -- (n,1) ndarray of our confidence that each sample belongs to its predicted class
	'''			
	
	# 1. TODO: figure out how many classes there are
	#print("prior shape" + str(priors.shape))
	class_labels, class_counts = np.unique( len(priors), return_counts=True )	# determine if there is more than one class label present
	num_classes = len(priors)
	#print(" classlabels" + str(class_labels) + " and " + str(class_counts))
	# 2. TODO: Use Bayes Rule to estimate the probability of each class label given each sample (row) in X
	# 			Pro Tip: Loop over a list of class names rather than the rows of X, vectorizing Bayes Rule within each class
	# loop over class_labels
	n = X.shape[0]
	#print(n)
	#print(num_classes)
	P_s_c = np.zeros((n, num_classes))
	P_c_s = np.zeros((n, num_classes))
	#print(P_s_c)
	for C in range (num_classes):
		#print("test")
		mu = means[C, :]
		sigma = variances[C, :]
		height = 1/ np.sqrt(2*np.pi * sigma)
		exponent = -0.5 * (X - mu)**2 / sigma
		P_s_c[:, C] =np.prod( height * np.exp(exponent), axis=1)
	P_c_s = P_s_c * priors
	print("P_s_c" + str(P_s_c))
	print("P_c_s" + str(P_c_s))
	# 3. TODO: Assign each sample to the class with the greatest probability
	C_pred = np.argmax(P_c_s, axis=1)
	P_pred = np.max(P_c_s, axis=1)
	return C_pred, P_pred


# MLVT students do NOT need to edit this function (but you can if you like)
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
	C_train_pred, _ = predict_nb( means, variances, priors, X_train )
	Conf_train, acc_train, _, _, _ = evaluate( X_train, C_train, C_train_pred, class_names, title+": NB Training Set")
	
	# Evaluate performance of the classifier on test set
	print("\n\nTEST SET PERFORMANCE:")
	C_test_pred, _ = predict_nb( means, variances, priors, X_test )
	Conf_test, acc_test, _, _, _ = evaluate( X_test, C_test, C_test_pred, class_names, title+": NB Test Set")

	# Visualize the classifier's output as a scatterplot projected onto the first 2 axes of X
	fig, ax = plt.subplots()
	cmap = cm.get_cmap( "viridis", num_classes ) 
	
	for c in range(num_classes):
		label = class_names[c]
		color = cmap( label/(num_classes-1) )
		print(c, label, color)
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

	return C_test_pred, Conf_test, acc_test, Conf_train, acc_train 
	

# MLVT students do NOT need to edit this function (but you can if you like)
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
	

# MLVT students do NOT need to edit this function (but you can if you like)
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


# MLVT students do NOT need to edit this function (but you can if you like)
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
	x = X_train[ :, col_x ].flatten()
	y = X_train[ :, col_y ].flatten()
	color_true = cmap( C_train/(num_classes-1) )
	#color_pred = cmap( C_train_pred/(num_classes-1) )
	ax.scatter( x, y, marker='o', s=35, c=color_true, edgecolor='k', alpha=0.50, label="train" )
	incorrect = C_train.flatten() != C_train_pred.flatten()
	ax.plot( x[incorrect], y[incorrect], 'xr', markersize=10, linewidth=2, label="incorrect (training)" )
	
	# Visualize test set as diamonds
	x = X_test[ :, col_x ].flatten()
	y = X_test[ :, col_y ].flatten()
	#color_true = cmap( C_test/(num_classes-1) )
	color_pred = cmap( C_test_pred/(num_classes-1) )
	ax.scatter( x, y, marker='d', s=55, c=color_pred, edgecolor='k', alpha=1.0, label="test" )
	incorrect = C_test.flatten() != C_test_pred.flatten()
	ax.plot( x[incorrect], y[incorrect], '+r', markersize=12, linewidth=2, label="incorrect (test)" )
	ax.legend()

	return ax


# MLVT students do NOT need to edit this function (but you can if you like)
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
	color_true = cmap( C_train/(num_classes-1) )
	ax.scatter( x, np.zeros(x.shape), marker='o', s=35, c=color_true, edgecolor='k', alpha=0.50, label="train" )
	incorrect = C_train.flatten() != C_train_pred.flatten()
	ax.plot( x[incorrect], np.zeros(x[incorrect].shape), 'xr', markersize=10, linewidth=2, label="incorrect (training)" )
	
	# Visualize test set as diamonds
	x = X_test[ :, col_x ].flatten()
	color_pred = cmap( C_test_pred/(num_classes-1) )
	ax.scatter( x, np.zeros(x.shape), marker='d', s=55, c=color_pred, edgecolor='k', alpha=1.0, label="test" )
	incorrect = C_test.flatten() != C_test_pred.flatten()
	ax.plot( x[incorrect], np.zeros(x[incorrect].shape), '+r', markersize=12, linewidth=2, label="incorrect (test)" )
	ax.legend()

	return ax

# MLVT students do NOT need to edit this function (but you can if you like)
def main( argv ):
	''' Parse command line arguments: 
		-- argv[0] is always the name of the program run from the terminal, e.g. "classification.py"
		-- argv[1] should be the path of a data file (e.g. *.DATA or *.CSV), e.g. "..\\data\\iris_preproc.csv"
		-- argv[2] should be the name of the class label feature, e.g. "C=species"
		-- Additional arguments are optional: "K=7" to use 7 nearest neighbors in KNN, and any number of input features, e.g. "X=petal length (cm)"
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
		filepath = os.path.join(current_directory, "data", "iris_preproc.csv")

	# Read in the dataset and remove NaNs
	data, headers, title = vis.read_csv( filepath )
	data, headers = vis.remove_nans( data, headers )
	
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
				# if("K=?" in param):
				# 	k_param = int(param.split("?")[1])
				# 	k = list(range(1,k_param + 1))
				# 	print( f"List of distances K selected: {k}" )
				# else:
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


if __name__=="__main__":
	main( sys.argv )
	plt.show()