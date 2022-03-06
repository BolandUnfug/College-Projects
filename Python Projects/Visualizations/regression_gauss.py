from regression_poly import *

def build_input_matrix_gauss( X, centroids=[], widths=[] ):
	''' Builds a gaussian mixture model (gauss) input matrix for linear regression.

	INPUT
	X -- (n,m) ndarray, with a row for each datum and a column for each input feature
	centroids -- (k,m) ndarray containing the mean of each gaussian kernel, with a row for each gaussian kernel and a column for each input feature
	widths -- (m,) ndarray containing the standard deviation of the gaussian kernels along each input feature, with a row for each input feature

	OUTPUT
	A -- (n,m*kernels+1) ndarray, with a row for each datum and a column for each gaussian kernel, followed by a column of ones.
	'''	
	
	kernels = max( [len(centroids), len(widths), 5] )

	if len(widths)==0:
		# Stretch gaussian kernels along each input feature axis for even coverage
		ptp = np.ptp( X, axis=0 )
		widths = (ptp / kernels) * 1.0

	if len(centroids==0):
		# Distribute gaussian kernels evenly along each input feature axis
		xmin = np.min( X, axis=0 )
		xmax = np.max( X, axis=0 )
		centroids = np.linspace( xmin, xmax, kernels )
	
	# Compute the unnormalized probability of each sample with respect to each of its features' gaussian kernels
	# (No sense normalizing if we're just going to find new weights anyway)
	n = X.shape[0]							# number of input samples
	m = X.shape[1]							# number of input features
	A = np.ones( (n,m*kernels+1) )
	for k in range( kernels ):		# for each gaussian kernel...
		for feat in range( m ):		# ... raise each input feature to that power
			col = k*m + feat
			A[ :, col ] = np.exp( -(X[:,feat] - centroids[k,feat])**2 / (2*widths[feat]**2) )
	return A


def model_gauss( X, X_headers, Y, Y_header, kernels=5, title="Gaussian Model" ):
	''' Builds a gaussian regression model to fit Y using 1 or more X features.

	INPUT \n
		X -- (n,m) ndarray, with a row for each datum and a column for each input feature \n
		X_headers -- list of str, the names of all input feature
		Y -- (n,1) ndarray, with a row for each datum and 1 column for the target feature \n
		Y_header -- str, the name of the target feature
		kernels -- int, the number of gaussian kernels to use in the model \n
		title -- str, text that will appear at the top of the figure window
 	
	OUTPUT \n
		W -- (m*k+1,1) ndarray, with a row for each coefficient in the model \n
		r_sq_test -- float, the coefficient of determination withn the test set \n
		rmse_test -- float, the root mean squared error of predictions within the test set \n
		r_sq_train -- float, the coefficient of determination within the training set \n
		rmse_train -- float, the root mean squared error of predictions within the training set \n
	'''	
	# Distribute kernels evenly along each input axis
	xmin = np.min( X, axis=0 )
	xmax = np.max( X, axis=0 )
	centroids = np.linspace( xmin, xmax, kernels )
	ptp = np.ptp( X, axis=0 )
	widths = (ptp / kernels) * 1.0

	# Fit weights to polynomial basis functions
	X_train, Y_train, X_test, Y_test = partition( X, Y )
	A_train = build_input_matrix_gauss( X_train, centroids, widths  )
	A_test = build_input_matrix_gauss( X_test, centroids, widths  )
	W = train( A_train, Y_train )

	# Evaluate performance of the model on the training partition (the same samples used to calculate the weights W)
	Y_pred_train = predict( A_train, W )
	r_sq_train, rmse_train = evaluate( Y_pred_train, Y_train )

	# Detect overfitting by evaluating the model's performance on the test partition (samples that were withheld during training)
	Y_pred_test = predict( A_test, W )
	r_sq_test, rmse_test = evaluate( Y_pred_test, Y_test )
	
	# Display model weights and performance in the terminal
	print( "\n", title )
	m = X.shape[1]
	print( "Output feature: " + Y_header )
	print( "Input features: " + str(X_headers) )
	print( "Centroids:" )
	print( centroids )
	print( "Widths:", widths )
	print( "Intercept:", W[-1] )
	print( "Weights: W.T = ", W.T )
	print( "Performance:" )
	print( f"\tTraining Set: R^2 = {r_sq_train:0.3f}, RMSE = {rmse_train:0.3f}" )
	print( f"\tTest Set:     R^2 = {r_sq_test:0.3f}, RMSE = {rmse_test:0.3f}" )

	# Visualize the model and its residuals. If there is more than 1 input feature, project onto each X axis separately.
	n = X.shape[0]
	m = X.shape[1]
	fig, ax = plt.subplots( nrows=2, ncols=m, sharex='col', sharey='row', squeeze=False )
	title += f", K={kernels:d}\nTest R^2={r_sq_test:0.2f}, RMSE={rmse_test:0.2f}"
	plt.suptitle( title + "\n2D projection(s)"  )
	n_line = 100*kernels
	X_line = np.linspace( xmin, xmax, n_line )
	for j in range(m):
		# Project the model onto input axis X[:,j]
		X_line_j = np.ones( X_line.shape ) * np.mean( X, axis=0 ) #np.zeros( X_line.shape )
		X_line_j[:,j] = X_line[:,j]
		A_line_j = build_input_matrix_gauss( X_line_j, centroids, widths )
		Y_line_j = predict( A_line_j, W )

		# Visualize data + model
		ax[0,j].plot( X[:,j], Y, 'bo', alpha=0.25 )
		ax[0,j].plot( X_line[:,j], Y_line_j, '-m' )
		ax[0,j].set_ylabel( Y_header )
		ax[0,j].grid( True )

		# Visualize residuals
		ax[1,j].plot( X_train[:,j], np.abs(Y_pred_train-Y_train), 'kx', alpha=0.33, label="training set" )
		ax[1,j].plot( X_test[:,j], np.abs(Y_pred_test-Y_test), 'rx', alpha=1.0, label="test set" )
		ax[1,j].set_ylabel( "| y* - y |" )
		ax[1,j].set_xlabel( X_headers[j] )
		ax[1,j].grid( True )
	
	fig.tight_layout()
	plt.legend()
	return W, r_sq_test, rmse_test, r_sq_train, rmse_train


def model_gauss_surface( X, X_headers, Y, Y_header, kernels=5, title="Multiple Gaussian Regression" ):
	'''
	Use multiple gaussian regression to create a surface that fits the 2 input features in X 
	to the target feature in Y.

	ARGS
		X -- (n,2) ndarray containing 2 columns of independent features
		X_headers -- list of two strings containing the names of features in X
		Y -- (n,1) ndarray containing 1 column of dependent feature
		Y_header -- string, the name of the feature in Y
		kernels -- int, the number of gaussian kernels to use in the model
		title -- the title of the output figure

	RETURN
		None
	'''
	# Distribute kernels evenly along each input axis
	min = np.min( X, axis=0 )
	max = np.max( X, axis=0 )
	centroids = np.linspace( min, max, kernels )
	ptp = np.ptp( X, axis=0 )
	widths = (ptp / kernels) * 1.0

	# Fit weights to gaussian basis functions
	X_train, Y_train, X_test, Y_test = partition( X, Y )
	A_train = build_input_matrix_gauss( X_train, centroids, widths  )
	A_test = build_input_matrix_gauss( X_test, centroids, widths  )
	W = train( A_train, Y_train )
	k = A_test.shape[1]

	# Evaluate performance of the model on the training partition (the same samples used to calculate the weights W)
	Y_pred_train = predict( A_train, W )
	r_sq_train, rmse_train = evaluate( Y_pred_train, Y_train )

	# Detect overfitting by evaluating the model's performance on the test partition (samples that were withheld during training)
	Y_pred_test = predict( A_test, W )
	r_sq_test, rmse_test = evaluate( Y_pred_test, Y_test )

	# Build a mesh that follows the shape of the surface
	n_line = 100*kernels
	x0_line = np.linspace(np.min(X[:,0]),np.max(X[:,0]), n_line)
	x1_line = np.linspace(np.min(X[:,1]),np.max(X[:,1]), n_line)
	X0_mesh, X1_mesh = np.meshgrid( x0_line, x1_line )
	n_mesh = X0_mesh.shape[0] * X0_mesh.shape[1]
	X_mesh = np.hstack((X0_mesh.reshape(n_mesh,1), X1_mesh.reshape(n_mesh,1)))
	A_mesh = build_input_matrix_gauss( X_mesh, centroids, widths )
	Y_pred_mesh = (A_mesh @ W).reshape( X0_mesh.shape )

	# Display model weights and performance in the terminal
	print( "\n", title )
	m = X.shape[1]
	print( "Output feature: " + Y_header )
	print( "Input features: " + str(X_headers) )
	print( "Centroids:" )
	print( centroids )
	print( "Widths:", widths )
	print( "Intercept:", W[-1] )
	print( "Weights: W.T = ", W.T )
	print( "Performance:" )
	print( f"\tTraining Set: R^2 = {r_sq_train:0.3f}, RMSE = {rmse_train:0.3f}" )
	print( f"\tTest Set:     R^2 = {r_sq_test:0.3f}, RMSE = {rmse_test:0.3f}" )

	# Visualize the surface
	fig, ax = plt.subplots( subplot_kw={"projection": "3d"} )
	title += f", K={kernels:d}\nTest R^2 = {r_sq_test:.3f}, RMSE = {rmse_test:.3f}"
	plt.suptitle( title )
	ax.plot_surface(X0_mesh, X1_mesh, Y_pred_mesh, antialiased=False, alpha=0.25 )
	ax.scatter( X_train[:,0], X_train[:,1], Y_train, c='b', alpha=0.5, label="training set" )
	ax.scatter( X_test[:,0], X_test[:,1], Y_test, c='r', alpha=0.5, label="test set" )
	ax.set_xlabel( X_headers[0] )
	ax.set_ylabel( X_headers[1] )
	ax.set_zlabel( Y_header )
	ax.grid( True )
	ax.legend()
	fig.tight_layout()
	return W, r_sq_test, rmse_test


def model_gauss_pairwise( data, headers, kernels=5, title="Single Gaussian Regression" ):
	n = data.shape[0]
	m = data.shape[1]
	fig, ax = plt.subplots( nrows=m, ncols=m, sharex='col', sharey='row', squeeze=False )
	plt.suptitle( title )	
	print( "\n" + title)
	for row in range(m):
		ax[row,0].set_ylabel( headers[row] )
		for col in range(m):
			if row == m-1:
				ax[row,col].set_xlabel( headers[col] )

			# Solve for the weights that define the line of best fit and evaluate the
			# "goodness" of fit with the coefficient of determination, r_squared
			X = data[:,col].reshape((n,1))
			Y = data[:,row].reshape((n,1))
			
			# Distribute kernels evenly along each input axis
			min = np.min( X, axis=0 )
			max = np.max( X, axis=0 )
			centroids = np.linspace( min, max, kernels )
			ptp = np.ptp( X, axis=0 )
			widths = (ptp / kernels) * 1.0

			# Fit weights to polynomial basis functions
			X_train, Y_train, X_test, Y_test = partition( X, Y )
			A_train = build_input_matrix_gauss( X_train, centroids, widths  )
			A_test = build_input_matrix_gauss( X_test, centroids, widths  )
			W = train( A_train, Y_train )

			# Measure model performance
			Y_pred_train = predict( A_train, W )
			r_sq_train, rmse_train = evaluate( Y_pred_train, Y_train )
			Y_pred_test = predict( A_test, W )
			r_sq_test, rmse_test = evaluate( Y_pred_test, Y_test )
			
			# Display model weights and performance in the terminal
			print( "\nOutput feature: " + headers[row] )
			print( "Input features: " + headers[col] )
			print( "Centroids:" )
			print( centroids )
			print( "Widths:", widths )
			print( "Intercept:", W[-1] )
			print( "Weights: W.T = ", W.T )
			print( "Performance:" )
			print( f"\tTraining Set: R^2 = {r_sq_train:0.3f}, RMSE = {rmse_train:0.3f}" )
			print( f"\tTest Set:     R^2 = {r_sq_test:0.3f}, RMSE = {rmse_test:0.3f}" )
			
			# Visualize predictions Y_line at evenly spaced points along a line
			n_line = 100
			X_line = np.linspace( np.min(X), np.max(X), n_line ).reshape((n_line,1))
			A_line = build_input_matrix_gauss( X_line, centroids, widths )
			Y_line = predict( A_line, W )
			ax[row,col].scatter( X, Y, c='b', alpha=0.33 )
			ax[row,col].plot( X_line, Y_line, '-m' )
			ax[row,col].set_title( f"R^2={r_sq_test:0.3f}" )
			ax[row,col].grid( True )

	fig.tight_layout()
	return fig


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
	data, headers = vis.remove_nans( data, headers )
	
	# Let the user name a polynomial degree, number of gaussian kernels, target feature (Y axis), and/or up to input features (X axis) using the commandline 
	# parameter prefixes "D=" for polynomial degree, "K=" for number of gaussian kernels, "Y=" for target feature, "X=" for each input feature.
	# By default it will fit a line to each projection in the dataset's pair plot.
	Y_header = ""
	Y = None
	X_headers = []
	X = None
	kernels = 5
	n = data.shape[0]
	if len(argv) > 2:
		for param in argv[2:]:

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

			elif "K=" in param:	
				# Let the user specify the number of gaussian kernels that they want to use in the model with the syntax "K=2", where 2 could be any integer
				K_param = param.split("=")[1]
				kernels = int( K_param )
				print( f"Number of gaussian kernels K selected: {kernels}" )

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
	
	if len(Y_header) > 0:
		# Include regression model descriptors in the title
		if X.shape[1] > 1:
			title += " Multiple"
		else:
			title += " Single"
		title += " Gaussian Regression"

		# Fit a multiple regression model to all available X features using a gaussian mixture model
		model_gauss( X, X_headers, Y, Y_header, kernels=kernels, title=title ) 

		# Fit a polynomial surface to each two pairs of X features
		if X.shape[1] > 1:
			for x0 in range(X.shape[1]-1):
				for x1 in range(x0+1,X.shape[1]):
					model_gauss_surface( X[:,[x0,x1]], [X_headers[x0], X_headers[x1]], Y, Y_header, kernels=kernels, title=title )

	else:
		# Fit pairwise single regression models with gaussian basis functions with each possible X-Y feature pair
		model_gauss_pairwise( data, headers, kernels=kernels, title=title+" Pairwise Single Gaussian Regression")



if __name__=="__main__":
	main( sys.argv )
	plt.show()