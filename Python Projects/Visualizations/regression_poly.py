import numpy as np
import matplotlib.pyplot as plt 
import os
import sys
import visualization as vis

def partition( X, Y, pct_train=0.70 ):
	'''
	Partition all available samples into separate training and test sets in order to prevent overfitting.

	ARGS
		X -- (n,m) ndarray, with a row for each datum and a column for each input feature
		Y -- (n,1) ndarray, with a row for each datum's known target output
		pct_train -- float, the percent of all samples to include in the training set. The rest will be used to test.

	RETURN
		X_train -- (n_train,m) ndarray, with a row for each datum in the training set and a column for each input feature
		Y_train -- (n_train,1) ndarray, containing the known output for each input (row) in A_train
		X_test -- (n_test,m) ndarray, with a row for each datum in the test set and a column for each input feature
		Y_test -- (n_test,1) ndarray, containing the known output for each input (row) in A_test
	'''
	# create indices, a set of positions
	indices = np.arange(X.shape[0]) 
	np.random.shuffle(indices) # shuffle the positions

	X = X[indices] # set X to the new positions
	Y = Y[indices] # set Y to the same new positions

	split = int(len(X)*pct_train) # since its shuffled, split at the percentage

	print("location:" + str(split)) # shows where its split

	# everything below the split
	X_train = X[:split] 
	Y_train = Y[:split]
	# everything above the split
	X_test = X[split:]
	Y_test = Y[split:] 

	return X_train, Y_train, X_test, Y_test


def train( A, Y ):
	''' Trains a regression model to find weights for the columns of A (which could include any basis functions) that best predict the target Y.

	INPUT
		A -- (n,d) ndarray, with a row for each datum and a column for each basis function
		Y -- (n,1) ndarray, with a row for each datum and 1 column for the target feature

	OUTPUT
		W -- (m*d+1,1) ndarray, with a row for each coefficient in the model
	'''
	W = np.linalg.pinv( A.T @ A ).T @ A.T @ Y
	return W


def predict( A, W ):
	''' Produces predictions Y_pred using input matrix A (whose columns could represent any basis functions) and weights W.

	ARGS
		A -- (n,k) ndarray, with a row for each datum and a column for each basis function
		W -- (k,1) ndarray, with a row for each basis function's weight

	RETURN
		Y_pred -- (n,1) ndarray, containing the predicted output for each input (row) in A
	'''
	Y_pred = A @ W
	return Y_pred


def evaluate( Y_pred, Y ):
	''' Evaluates the performance of a regression model that has produced predictions Y_pred for targets Y.

	ARGS
		Y_pred -- (n,1) ndarray, containing the model's predictions of targets Y
		Y      -- (n,1) ndarray, containing known targets that map to the same inputs that were used to make predictions Y_pred

	RETURN
		r_squared -- float, the R^2 coefficient of determination of Y_pred with respect to Y
		rmse -- the root mean squared error of Y_pred with respect to Y
	'''
	n = Y.shape[0]								# number of samples
	R = Y - Y_pred								# residuals
	rss = R.T @ R								# residual sum of squares
	ss = (Y-np.mean(Y)).T @ (Y - np.mean(Y))	# sum of squares
	r_squared = 1 - rss / ss					# R^2 coefficient of determination
	if type(r_squared) == np.ndarray:
		r_squared = r_squared[0,0]
	rmse = np.sqrt( rss / n )					# root mean squared error
	if type(rmse) == np.ndarray:
		rmse = rmse[0,0]
	return r_squared, rmse


def build_input_matrix_poly( X, degree=1 ):
	''' Builds a polynomial input matrix for linear regression.

	INPUT
	X -- (n,m) ndarray, with a row for each datum and a column for each input feature
	degree -- int, the maximum polynomial degree in the completed input matrix

	OUTPUT
	A -- (n,m*degree+1) ndarray, with a row for each datum and a column for X raised to each power from "degree" (column 0,
			on the far left) through 1 (column -2), followed by a column of ones (column -1, on the far right).
	'''	
	n = X.shape[0]							# number of input samples
	m = X.shape[1]							# number of input features
	A = np.ones( (n,m*degree+1) )
	for d in range( degree, 0, -1 ):		# for each polynomial degree...
		for feat in range( 0, m, 1 ):		# ... raise each input feature to that power
			col = (degree-d)*m + feat
			A[ :, col ] = X[:,feat]**d
	return A


def model_poly( X, X_headers, Y, Y_header, degree=1, title="Polynomial Model", draw=True ):
	''' Builds a polynomial regression model to fit Y using 1 or more X features.

	INPUT \n
		X -- (n,m) ndarray, with a row for each datum and a column for each input feature \n
		X_headers -- list of str, the names of all input feature
		Y -- (n,1) ndarray, with a row for each datum and 1 column for the target feature \n
		Y_header -- str, the name of the target feature
		degree -- int, the maximum polynomial degree (exponential power) in the completed input matrix \n
		title -- str, text that will appear at the top of the figure window
 	
	OUTPUT \n
		W -- (m*d+1,1) ndarray, with a row for each coefficient in the model \n
		r_sq_test -- float, the coefficient of determination withn the test set \n
		rmse_test -- float, the root mean squared error of predictions within the test set \n
		r_sq_train -- float, the coefficient of determination within the training set \n
		rmse_train -- float, the root mean squared error of predictions within the training set \n
	'''	

	# Fit weights to polynomial basis functions
	X_train, Y_train, X_test, Y_test = partition( X, Y )
	A_train = build_input_matrix_poly( X_train, degree )
	A_test  = build_input_matrix_poly( X_test,  degree )
	W = train( A_train, Y_train )

	# Evaluate performance of the model on the training partition (the same samples used to calculate the weights W)
	Y_pred_train = predict( A_train, W )
	r_sq_train, rmse_train = evaluate( Y_pred_train, Y_train )

	# Detect overfitting by evaluating the model's performance on the test partition (samples that were withheld during training)
	Y_pred_test = predict( A_test, W )
	r_sq_test, rmse_test = evaluate( Y_pred_test, Y_test )

	# Display model weights and performance in the terminal
	if(draw == True):
		print( "\n", title )
		m = X.shape[1]
		basis = "Basis Functions: ["
		model = "Model: "
		for d in range(degree, 0, -1):
			for feat in range(m):
				w = W[(degree-d)*m + feat]
				basis += f"({X_headers[feat]})^{d},  "
				model += f"{w}*({X_headers[feat]})^{d}  +  "
		basis += "1"
		model += f"{w}"
		print( "Output feature: " + Y_header )
		print( "Input features: " + str(X_headers) )
		print( basis )
		print( f"Weights: W.T = {W.T}" )
		print( model )
		print( "Performance:" )
		print( f"\tTraining Set: R^2 = {r_sq_train:0.3f}, RMSE = {rmse_train:0.3f}" )
		print( f"\tTest Set:     R^2 = {r_sq_test:0.3f}, RMSE = {rmse_test:0.3f}" )

		# Visualize the model and its residuals. If there is more than 1 input feature, project onto each X axis separately.
		n = X.shape[0]
		m = X.shape[1]
		fig, ax = plt.subplots( nrows=2, ncols=m, sharex='col', sharey='row', squeeze=False )
		title += f", D={degree:d}\nTest R^2={r_sq_test:0.2f}, RMSE={rmse_test:0.2f}"
		title += f"\nW.T={W.T}"
		plt.suptitle( title + "\n2D projection(s)"  )
		xmin = np.min(X, axis=0)
		xmax = np.max(X, axis=0)
		n_line = 100*degree
		X_line = np.linspace( xmin, xmax, n_line )
		for j in range(m):
			# Project the model onto input axis X[:,j]
			X_line_j = np.ones( X_line.shape ) * np.mean( X, axis=0 ) #np.zeros( X_line.shape )
			X_line_j[:,j] = X_line[:,j]
			A_line_j = build_input_matrix_poly( X_line_j, degree )
			Y_line_j = predict( A_line_j, W )

			# data + model
			ax[0,j].plot( X[:,j], Y, 'bo', alpha=0.25 )
			ax[0,j].plot( X_line[:,j], Y_line_j, '-m' )
			ax[0,j].set_ylabel( Y_header )
			ax[0,j].grid( True )

			# residuals
			ax[1,j].plot( X_train[:,j], np.abs(Y_pred_train-Y_train), 'kx', alpha=0.33, label="training set" )
			ax[1,j].plot( X_test[:,j], np.abs(Y_pred_test-Y_test), 'rx', alpha=1.0, label="test set" )
			ax[1,j].set_ylabel( "| y* - y |" )
			ax[1,j].set_xlabel( X_headers[j] )
			ax[1,j].grid( True )
	
		fig.tight_layout()
		plt.legend()
	return W, r_sq_test, rmse_test, r_sq_train, rmse_train


def model_poly_surface( X, X_headers, Y, Y_header, degree=1, title="Multiple Polynomial Regression", draw = True ):
	'''
	Use multiple polynomial regression to create a surface that fits the 2 input features in X 
	to the target feature in Y.

	ARGS
		X -- (n,2) ndarray containing 2 columns of independent features
		X_headers -- list of two strings containing the names of features in X
		Y -- (n,1) ndarray containing 1 column of dependent feature
		Y_header -- string, the name of the feature in Y
		degree -- int, the maximum polynomial degree to use in the model
		title -- the title of the output figure

	RETURN
		W, r_sq_test, rmse_test
	'''
	# Fit weights to polynomial basis functions
	X_train, Y_train, X_test, Y_test = partition( X[:,0:2], Y )
	A_train = build_input_matrix_poly( X_train, degree )
	A_test  = build_input_matrix_poly( X_test,  degree )
	W = train( A_train, Y_train )
	k = A_train.shape[1]

	# Evaluate performance of the model on the training partition (the same samples used to calculate the weights W)
	Y_pred_train = predict( A_train, W )
	r_sq_train, rmse_train = evaluate( Y_pred_train, Y_train )

	# Detect overfitting by evaluating the model's performance on the test partition (samples that were withheld during training)
	Y_pred_test = predict( A_test, W )
	r_sq_test, rmse_test = evaluate( Y_pred_test, Y_test )

	# Build a mesh that follows the shape of the surface
	n_line = 100*degree
	x0_line = np.linspace(np.min(X[:,0]),np.max(X[:,0]), n_line)
	x1_line = np.linspace(np.min(X[:,1]),np.max(X[:,1]), n_line)
	X0_mesh, X1_mesh = np.meshgrid( x0_line, x1_line )
	n_mesh = X0_mesh.shape[0] * X0_mesh.shape[1]
	X_mesh = np.hstack((X0_mesh.reshape(n_mesh,1), X1_mesh.reshape(n_mesh,1)))
	A_mesh = build_input_matrix_poly( X_mesh, degree )
	Y_pred_mesh = (A_mesh @ W).reshape( X0_mesh.shape )
	if (draw == True):
		# Display model weights and performance in the terminal
		print( "\n", title )
		m = X.shape[1]
		basis = "Basis Functions: ["
		model = "Model: "
		for d in range(degree, 0, -1):
			for feat in range(2):
				w = W[(degree-d)*m + feat]
				basis += f"({X_headers[feat]})^{d},  "
				model += f"{w}*({X_headers[feat]})^{d}  +  "
		basis += "1"
		model += f"{w}"
		print( "Output feature: " + Y_header )
		print( "Input features: " + str(X_headers) )
		print( basis )
		print( f"Weights: W.T = {W.T}" )
		print( model )
		print( "Performance:" )
		print( f"\tTraining Set: R^2 = {r_sq_train:0.3f}, RMSE = {rmse_train:0.3f}" )
		print( f"\tTest Set:     R^2 = {r_sq_test:0.3f}, RMSE = {rmse_test:0.3f}" )

		# Visualize the surface
		fig, ax = plt.subplots( subplot_kw={"projection": "3d"} )
		title += f", D={degree:d}\nTest R^2 = {r_sq_test:.3f}, RMSE = {rmse_test:.3f}"
		plt.suptitle( title )
		ax.plot_surface(X0_mesh, X1_mesh, Y_pred_mesh, antialiased=False, alpha=0.25 )
		#ax.scatter( X[:,0], X[:,1], Y )
		ax.scatter( X_train[:,0], X_train[:,1], Y_train, c='b', alpha=0.5, label="training set" )
		ax.scatter( X_test[:,0], X_test[:,1], Y_test, c='r', alpha=0.5, label="test set" )
		ax.set_xlabel( X_headers[0] )
		ax.set_ylabel( X_headers[1] )
		ax.set_zlabel( Y_header )
		ax.grid( True )
		ax.legend()
		fig.tight_layout()
	return W, r_sq_test, rmse_test


def model_poly_pairwise(data, headers, degree=1, title="Single Polynomial Regression", draw=True ):
	n = data.shape[0]
	m = data.shape[1]
	if(draw == True):
		fig, ax = plt.subplots( nrows=m, ncols=m, sharex='col', sharey='row', squeeze=True )
		plt.suptitle( title )
		print( "\n", title )
		for row in range(m):
			ax[row,0].set_ylabel( headers[row] )
			for col in range(m):
				if row == m-1:
					ax[row,col].set_xlabel( headers[col] )

				# Solve for the weights that define the line of best fit and evaluate the
				# "goodness" of fit with the coefficient of determination, r_squared
				X = data[:,col].reshape((n,1))
				Y = data[:,row].reshape((n,1))

				# Fit weights to polynomial basis functions
				X_train, Y_train, X_test, Y_test = partition( X, Y )
				A_train = build_input_matrix_poly( X_train, degree )
				A_test  = build_input_matrix_poly( X_test,  degree )
				W = train( A_train, Y_train )

				# Evaluate performance of the model on the training partition (the same samples used to calculate the weights W)
				Y_pred_train = predict( A_train, W )
				r_sq_train, rmse_train = evaluate( Y_pred_train, Y_train )

				# Detect overfitting by evaluating the model's performance on the test partition (samples that were withheld during training)
				Y_pred_test = predict( A_test, W )
				r_sq_test, rmse_test = evaluate( Y_pred_test, Y_test )
				
				# Display model weights and performance in the terminal
				basis = "\nBasis Functions: ["
				model = "Model: "
				for d in range(degree, 0, -1):
					w = W[degree-d]
					basis += f"({headers[col]})^{d},  "
					model += f"{w}*({headers[col]})^{d}  +  "	
				basis += "1"
				model += f"{w}"
				print( "Output feature: " + headers[row] )
				print( "Input features: " + headers[col] )
				print( basis )
				print( f"Weights: W.T = {W.T}" )
				print( model )
				print( "Performance:" )
				print( f"\tTraining Set: R^2 = {r_sq_train:0.3f}, RMSE = {rmse_train:0.3f}" )
				print( f"\tTest Set:     R^2 = {r_sq_test:0.3f}, RMSE = {rmse_test:0.3f}" )		

				# Visualize predictions Y_line at evenly spaced points along a line
				n_line = 100
				X_line = np.linspace( np.min(X), np.max(X), n_line ).reshape((n_line,1))
				A_line = build_input_matrix_poly( X_line, degree )
				Y_line = predict( A_line, W )
				ax[row,col].scatter( X, Y, c='b', alpha=0.33 )
				ax[row,col].plot( X_line, Y_line, '-m' )
				ax[row,col].set_title( f"R^2={r_sq_test:0.3f}" )
				ax[row,col].grid( True )

		fig.tight_layout()
		return fig
	else:
		for row in range(m):
			for col in range(m):
				# Solve for the weights that define the line of best fit and evaluate the
				# "goodness" of fit with the coefficient of determination, r_squared
				X = data[:,col].reshape((n,1))
				Y = data[:,row].reshape((n,1))

				# Fit weights to polynomial basis functions
				X_train, Y_train, X_test, Y_test = partition( X, Y )
				A_train = build_input_matrix_poly( X_train, degree )
				A_test  = build_input_matrix_poly( X_test,  degree )
				W = train( A_train, Y_train )

				# Evaluate performance of the model on the training partition (the same samples used to calculate the weights W)
				Y_pred_train = predict( A_train, W )
				r_sq_train, rmse_train = evaluate( Y_pred_train, Y_train )

				# Detect overfitting by evaluating the model's performance on the test partition (samples that were withheld during training)
				Y_pred_test = predict( A_test, W )
				r_sq_test, rmse_test = evaluate( Y_pred_test, Y_test )
				print(r_sq_test)
				return r_sq_test



def main( argv ):
	''' Parse command line arguments: 
		-- argv[0] is always the name of the program run from the terminal
		-- argv[1] should be the path of a data file (e.g. *.DATA or *.CSV) 
		-- argv[2] should be the name of the target feature that the user wants to predict (i.e. the Y-axis header)
		-- argv[3] should be whisper, where it only draws the graph, andsilent, where it draws nothing.
	'''

	# Since we don't care too much about decimal places, today, let's make the
	# output a little more human friendly. Heads up: This isn't always a good idea!
	np.set_printoptions(precision=3, suppress=True)
	draw = True
	# Determne the input file's path: either it was supplied in the commandline, or we should use iris as a default
	if len(argv) > 1:
		filepath = argv[1].strip()
	else:
		#filepath = "../data/iris_preproc.csv"
		current_directory = os.path.dirname(__file__)
		filepath = os.path.join(current_directory, "..", "data", "iris_preproc.csv")

	# Read in the dataset and remove NaNs
	data, headers, title = vis.read_csv( filepath )
	data, headers = vis.remove_nans( data, headers )
	
	# Let the user name a polynomial degree, number of gaussian kernels, target feature (Y axis), and/or up to input features (X axis) using the commandline 
	# parameter prefixes "D=" for polynomial degree, "Y=" for target feature, "X=" for each input feature.
	# By default it will fit a line to each projection in the dataset's pair plot.
	Y_header = ""
	Y = None
	X_headers = []
	X = None
	degree = [1]
	n = data.shape[0]

	if len(argv) > 2:
		for param in argv[2:]:
			if "whisper" in param:
				draw = False
			if "Y=" in param:
				# Find the column that contains the user's chosen feature name 
				Y_param = param.split("=")[1]
				if Y_param in headers:
					Y_idx = headers.index( Y_param )	# CAUTION: this will crash if the user's input does not match any element in the list of headers
					Y_header = headers[ Y_idx ]
					Y = data[ :, Y_idx ].reshape((n,1))
					print( f"Target Y feature selected: {Y_header}" )
				else:
					print( f"\nWARNING: '{Y_param}' not found in the headers list: {headers}. No target Y feature selected.\n" )

			elif "D=" in param:	
				if "D=?" in param:
					D_param = int(param.split("?")[1])
					degree = list(range(1,D_param + 1))
					print( f"List of Polynomial degrees D selected: {degree}" )
					draw = False
				else:
					# Let the user specify the maximum polynomial degree that they want to use in the model with the syntax "D=2", where 2 could be any integer
					D_param = param.split("=")[1]
					degree = [int( D_param )]
					print( f"Polynomial degree D selected: {degree}" )

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

	# If the user has specified a Y feature but no X features, use every feature except Y as input, 1 at a time
	if len(Y_header) > 0 and len(X_headers)==0:
		X = np.delete( data, Y_idx, axis=1 )
		X_headers = headers[:]
		X_headers.pop( Y_idx ) 

	# If the user has specified both X and Y features, we can ignore the rest of the dataset
	elif len(Y_header) > 0 and len(X_headers) > 0:
		data = np.hstack((X,Y))
		headers = X_headers + [ Y_header ]
	rsqlist = []
	for i in degree:
		if len(Y_header) > 0:
			
			# Include regression model descriptors in the title
			if X.shape[1] > 1:
				title += " Multiple"
			else:
				title += " Single"
			title += " Polynomial Regression"
		
			# Fit a multiple regression model to all available X features using polynomial basis functions
			W, rsq_test, rmse_test, rsq_train, rmse_test = model_poly( X, X_headers, Y, Y_header, degree=i, title=title, draw=draw ) 
			rsqlist.append(rsq_test)
			# Fit a polynomial surface to each two pairs of X features
			if X.shape[1] > 1:
				for x0 in range(X.shape[1]-1):
					for x1 in range(x0+1,X.shape[1]):
						W, rsq_test, rmse_test = model_poly_surface( X[:,[x0,x1]], [X_headers[x0], X_headers[x1]], Y, Y_header, degree=i, title=title, draw=draw )
						rsqlist.append(rsq_test)
		
		else:
			# Fit pairwise single regression models with polynomial basis functions with each possible X-Y feature pair
			if(draw == False):
				rsq_test = model_poly_pairwise( data, headers, degree=i, title=title+" Pairwise Single Polynomial Regression", draw=draw)
				rsqlist.append(rsq_test)
			else:
				model_poly_pairwise( data, headers, degree=i, title=title+" Pairwise Single Polynomial Regression", draw=draw)
			
	if (draw==False): plt.plot(degree, rsqlist)


if __name__=="__main__":
	main( sys.argv )
	plt.show()