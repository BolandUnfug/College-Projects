'''
pca.py

Implements Principal Components Analysis (PCA).

Execution:		>> python3 pca.py [path_to_dataset, str] "C=[class_column_header, str]" D=[number_neighbors, int]
	Examples:		>> python3 pca.py ../data/iris.csv D=3
					>> python3 pca.py ../data/iris.csv "C=species" D=2

Requires visualization.py in the same folder

@author Boland Unfug
@date March 3rd 2022
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import visualization as vis

def z_transform( X ):
	''' Normalize the dataset X by Z-score: subtract the mean and divide by the standard deviation.

	ARGS:
	X -- (n,m) ndarray of raw data, assumed to contain 1 row per sample and 1 column per feature.

	RETURNS:
	X_norm -- (n,m) ndarray of Z-score normalized data
	X_mean -- (m,) ndarray of the means of the features (columns) of the raw dataset X
	X_std -- (m,) ndarray of the standard deviations of the features (columns) of the raw dataset X
	'''
	#X_norm, X_mean, X_std = None, None, None	# TODO: Remove this line after implementing the Z transform
	X_mean = X.mean(axis=0)
	X_std = X.std(axis=0)
	X_norm = (X - X_mean) / X_std
	return X_norm, X_mean, X_std


def pca_cov( X ):
	"""Perform Principal Components Analysis (PCA) using the covariance matrix to identify principal components 
	(eigenvectors) and their scaled eigenvalues (which measure how much information each PC represents).
	
	INPUT:
	X -- (n,m) ndarray representing the dataset (observations), assuming one datum per row and one column per feature. 
			Must already be centered, so that the mean is at zero. Usually Z-score normalized. 
	
	OUTPUT:
	Y -- (n,m) ndarray representing rotated dataset (Y), 
	P -- (m,m) ndarray representing principal components (columns of P), a.k.a. eigenvectors
	e_scaled -- (m,) ndarray of scaled eigenvalues, which measure info retained along each corresponding PC """
	
	#Y, P, e_scaled = None, None, None	# TODO: Remove this after implementing pca_cov
	
	# TODO: Pull principal components and eigenvalues from covariance matrix
	C = np.cov( X, rowvar = False)
	(eigvals, P) = np.linalg.eig(C)
	
	# TODO: Sort principal components in order of descending eigenvalues
	order = np.argsort(eigvals)[::-1] #creats a list of ordered coordinates
	eigvals = eigvals[order] # sorts eigvalues by this order
	P = P[:,order] # sorts P by this order, so the coordinates still match up

	
	# TODO: Scale eigenvalues to calculate the percent info retained along each PC
	e_scaled = eigvals/ eigvals.sum()
		
	# TODO: Rotate data onto the principal components
	Y = X @ P

	
	return (Y, P, e_scaled)


def pca_svd( X ):
	"""Perform Principal Components Analysis (PCA) using singular value decomposition to identify principal components 
	(eigenvectors) and their scaled eigenvalues (which measure how much information each PC represents).
	
	INPUT:
	X -- (n,m) ndarray representing the dataset (observations), assuming one datum per row and one column per feature. 
			Must already be centered, so that the mean is at zero. Usually also Z-score normalized. 
	
	OUTPUT:
	Y -- (n,m) ndarray representing rotated dataset (Y), 
	P -- (m,m) ndarray representing principal components (columns of P), a.k.a. eigenvectors
	e_scaled -- (m,) ndarray of scaled eigenvalues, which measure info retained along each corresponding PC """
	
	#Y, P, e_scaled = None, None, None	# TODO: Remove this after implementing pca_svd
	
	# TODO: Pull principal components and eigenvalues from SVD
	(U, W, Vt ) = np.linalg.svd( X )
	eigvals = W**2
	P = Vt.T
	
	# TODO: Sort principal components in order of descending eigenvalues
	order = np.argsort(eigvals)[::-1] #creats a list of ordered coordinates
	eigvals = eigvals[order] # sorts eigvalues by this order
	P = P[:,order] # sorts P by this order, so the coordinates still match up
	
	# TODO: Scale eigenvalues to calculate the percent info retained along each PC
	e_scaled = eigvals/ eigvals.sum()
		
	# TODO: Rotate data onto the principal components
	Y = X @ P

	
	
	
	return (Y, P, e_scaled)


def reconstruct( Y, P, X_mean, X_std ):
	'''Reconstruct an approximation (X_rec) of the original dataset (X) by uncompressing the projection (Y).

	ARGS:
	Y -- (n,d) ndarray representing rotated and projected dataset (Y), 
	P -- (m,m) ndarray representing principal components (columns of P), a.k.a. eigenvectors
	X_mean -- (1,m) ndarray representing the mean of the original input features
	X_std -- (1,m) ndarray representing the standard deviation of the original input features
	
	RETURNS:
	X_rec -- (n,m) ndarray representing the reconstructed dataset. 
	'''
	#X_rec = None	#TODO: Remove this line when you've implemented reconstruction

	# TODO: Undo projection by padding Y with zeros (adding columns of zeros to the righthand side)
	
	print(Y.shape)
	print(P.shape)
	d = Y.shape[1] # this is a temp value, for some reason d is never given here.
	X_rec = (Y @ P[:,0:d].T)*X_std + X_mean
	# TODO: Undo rotatation using P

	# TODO: Undo scale using X_std
	
	# TODO: Undo translation using X_mean
	return X_rec


def scree_plot( eigenvals ):
	"""Visualize information retention per eigenvector.
	
	ARGS:	
	eigenvals -- (d,) ndarray of scaled eigenvalues.
	
	RETURNS:
	info_retention -- (d,) ndarray of accumulated information retained by multiple eigenvectors.  """
			
	# Visaulize individual information retention per eigenvector (eigenvalues)
	fig, ax = plt.subplots( 2, 1 )
	ax[0].plot( eigenvals, '-o', linewidth=2, markersize=5, markerfacecolor="w" )
	ax[0].set_ylim([-0.1, 1.1])
	ax[0].set_title( "Information retained by individual PCs" )
	ax[0].grid( True )
	
	# Visualize accumulated information retained by multiple eigenvectors
	info_retention = np.cumsum( eigenvals )
	ax[1].plot( info_retention, '-o', linewidth=2, markersize=5, markerfacecolor="w" )
	ax[1].set_ylim([-0.1, 1.1])
	ax[1].set_title( "Cumulative information retained by all PCs" )
	ax[1].grid( True )
	
	plt.pause(0.001)
		
	return info_retention


def pc_heatmap( P, e_scaled, X_headers ):
	''' Visualize principal components (eigenvectors) as a heatmap. 
	
	ARGS:
	P -- (m,m) ndarray of SORTED principal components (eigenvectors)
	e_scaled -- (m,) ndarray of scaled eigenvalues, which measure info retained along each corresponding PC
	X_headers -- list of strings that contain the name of each input feature in the original dataset
	
	RETURNS: 
	None
	'''
	
	# Generate the heatmap and colorbar
	fig, ax = plt.subplots()
	ax.set_title("Principal Components")
	im = ax.imshow( abs(P) )
	cbar = ax.figure.colorbar( im, ax=ax )

	# Annotate Y axis with the names of original features
	m = P.shape[0]
	step = 1
	if 10 < m:
		step = int(m/8)
	ticks = np.arange(0,m,step)
	ax.set_yticks( ticks )
	ax.set_yticklabels( X_headers[::step] )

	# Annotate X axis with the name of each PC and the info it retains
	pc_names = []
	for i in range(m):
		pc_names.append( f"P{i}\n{e_scaled[i]*100:.2f}%" )	
	ax.set_xticks( ticks )
	ax.set_xticklabels( pc_names[::step] )

	# Annotate cells with the floating point values of each PC element, unless this is such a high dimensional
	# dataset that they cells will be very closely packed and difficult to read
	if m <= 10:
		for feature in ticks:
			for pc in ticks:
				text = ax.text(pc, feature, f"{P[feature, pc]:.2f}", ha="center", va="center", color="w")

	fig.tight_layout()
	plt.pause(0.001)
	return None

def pca_processing (X, X_headers, d=2, C=None, C_header=None):
	''' Use PCA to project X onto a d-dimensional subspace defined by its first d principal components.

	ARGS:
	X -- (n,m) ndarray containing the original dataset's input features (WITHOUT class column, if applicable)
	X_headers -- list of strings that define the name of each input feature (column) in X
	d -- integer number of dimensions to project onto, must be <= m
	C -- optional (n,1) ndarray contianing column of class labels
	C_header -- optional string that defines the name of the class feature (column) in the original dataset

	RETURNS:
	Y -- (n,d) ndarray containing the compressed dataset rotated and projected onto the first d principal components.
	P -- (m,d) ndarray containing the first d principal components (eigenvectors), sorted in order of decreasing eigenvalues
	e_scaled -- (d,) ndarray containing the scaled eigenvalues (% info retained) of each PC
	'''

	# Normalize features by Z-score (so that features' units don't dominate PCs)
	var = np.var( X, axis=0 )
	if np.min(np.abs(var)) == 0.0:
		# there is at least one feature with zero variance ==> can center but cannot Z-transform
		X_mean = np.mean(X, axis=0)
		X_norm = X - X_mean
		X_std = np.ones(X_mean.shape)		
	else:
		# safe to Z-transform
		X_norm, X_mean, X_std = z_transform( X )

	# Choose PCA-COV or PCA-SVD based on dimensionality of the dataset
	n,m = X.shape
	if n > m:
		# more samples than features, so COV is safe.
		Y, P, e_scaled = pca_cov( X_norm )
	else:
		# more features than samples, so COV with break. Use SVD instead.
		Y, P, e_scaled = pca_svd( X_norm )

	# Project onto the first d PCs
	Y = Y[:,0:d]

	Y_headers = []
	for p in range(d):
		Y_headers.append( f"PC{p}\ne{p}={e_scaled[p]:.2f}" )
	
	# Sanity check: Print PCs and eigenvalues in the terminal
	print( "Eigenvectors (each column is a PC): \n\n", P, "\n" )
	print("\nScaled eigenvalues: \t", e_scaled, "\n" )

	return Y, P, e_scaled


	
def pca_analysis( X, X_headers, d=2, C=None, C_header=None ):
	''' Use PCA to project X onto a d-dimensional subspace defined by its first d principal components.

	ARGS:
	X -- (n,m) ndarray containing the original dataset's input features (WITHOUT class column, if applicable)
	X_headers -- list of strings that define the name of each input feature (column) in X
	d -- integer number of dimensions to project onto, must be <= m
	C -- optional (n,1) ndarray contianing column of class labels
	C_header -- optional string that defines the name of the class feature (column) in the original dataset

	RETURNS:
	Y -- (n,d) ndarray containing the compressed dataset rotated and projected onto the first d principal components.
	P -- (m,d) ndarray containing the first d principal components (eigenvectors), sorted in order of decreasing eigenvalues
	e_scaled -- (d,) ndarray containing the scaled eigenvalues (% info retained) of each PC
	'''

	# Visualize raw data
	if type(C) != np.ndarray:
		# No class label provided. Scatter all samples in the same color
		vis.scatter( X, X_headers, 0, 1, title="Original Dataset" )
		vis.heatmap( X, X_headers, title="Original Dataset" )
	else:
		# Sorting X by class label makes the heatmap easier to read
		class_order = np.argsort( C, axis=0 ).flatten()
		C = C[class_order, :]
		X = X[class_order, :]
		all_data = np.hstack(( X, C ))
		all_headers = X_headers + [C_header]
		vis.scatter( all_data, all_headers, 0, 1, -1, "Original Dataset" )
		vis.heatmap( all_data, all_headers, "Original Dataset" )
		
	# Normalize features by Z-score (so that features' units don't dominate PCs)
	var = np.var( X, axis=0 )
	if np.min(np.abs(var)) == 0.0:
		# there is at least one feature with zero variance ==> can center but cannot Z-transform
		X_mean = np.mean(X, axis=0)
		X_norm = X - X_mean
		X_std = np.ones(X_mean.shape)		
	else:
		# safe to Z-transform
		X_norm, X_mean, X_std = z_transform( X )
	
	# Choose PCA-COV or PCA-SVD based on dimensionality of the dataset
	n,m = X.shape
	if n > m:
		# more samples than features, so COV is safe.
		Y, P, e_scaled = pca_cov( X_norm )
	else:
		# more features than samples, so COV with break. Use SVD instead.
		Y, P, e_scaled = pca_svd( X_norm )

	# Project onto the first d PCs
	Y = Y[:,0:d]

	# Y's headers are the PC names and their scaled eigenvalues
	Y_headers = []
	for p in range(d):
		Y_headers.append( f"PC{p}\ne{p}={e_scaled[p]:.2f}" )
	
	# Sanity check: Print PCs and eigenvalues in the terminal
	print( "Eigenvectors (each column is a PC): \n\n", P, "\n" )
	print("\nScaled eigenvalues: \t", e_scaled, "\n" )
	
	# Visualize PCs with heatmap and cree plot
	info_retention = scree_plot( e_scaled )
	pc_heatmap( P, e_scaled, X_headers )

	# Visualize PCA data
	if type(C) != np.ndarray:
		# No class label provided. Scatter all samples in the same color
		vis.scatter( Y, Y_headers, 0, 1, title="2D PCA Projection" )
		vis.heatmap( Y, Y_headers, title=f"{d}-D PCA Projection" )
	else:
		all_data = np.hstack(( Y, C ))
		all_headers = Y_headers + [C_header]
		vis.scatter( all_data, all_headers, 0, 1, -1, "2D PCA Projection" )
		vis.heatmap( all_data, all_headers, title=f"{d}-D PCA Projection" )

	# Visualize the reconstruction
	X_rec = reconstruct( Y, P, X_mean, X_std )

	# RMSE of the reconstruction
	rmse = (np.sqrt(np.sum( (X - X_rec)**2, axis=1 ) / n)).flatten()[0]
	print( f"RMSE of {d}-D reconstruction = {rmse:.6f}")

	vis.scatter(X_rec, Y_headers, 0, 1, title = f"RMSE of {d}-D reconstruction = {rmse:.6f}" )

	return Y, P, e_scaled


def main( argv ):
	''' Parse command line arguments: 
		-- argv[0] is always the name of the program run from the terminal
		-- argv[1] should be the path of a data file (e.g. *.DATA or *.CSV) 
		-- any remaining args can be the number of PCs to project onto (e.g. "D=2") and/or the name of a class feature (e.g. "C=species")
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
	
	# Let the user name a class label feature (column header) to exclude from the principal components, as well as 
	# a number of dimensions D (integer) to project onto.
	d = 2
	n = data.shape[0]
	C_idx = None
	C_header = None
	C = None
	X_headers = headers[:]
	X = data[:,:]
	if len(argv) > 2:
		for param in argv[2:]:

			if "C=" in param:
				# Find the column that contains the user's chosen feature name 
				C_param = param.split("=")[1]
				if C_param in headers:
					C_idx = headers.index( C_param )	# CAUTION: this will crash if the user's input does not match any element in the list of headers
					C = data[ :, C_idx ].reshape((n,1))
					X = np.delete( data, C_idx, axis=1 )
					C_header = X_headers.pop( C_idx )
					print( f"Target class feature selected: {C_header}" )
				else:
					print( f"\nWARNING: '{C_param}' not found in the headers list: {headers}. No target class feature selected.\n" )

			elif "D=" in param:	
				# Let the user specify the number of PCs that they want to retain (project onto), e.g. "D=2", where 2 could be any integer
				d_param = param.split("=")[1]
				d = int( d_param )
				print( f"PCA configured to project onto first {d} PCs" )

	# Project onto first d PCs
	Y, P, e_scaled = pca_analysis( X, X_headers, d, C, C_header )


if __name__=="__main__":
	main( sys.argv )
	plt.show()