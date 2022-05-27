'''
cluster.py

Implements K-Means and Expectation Maximization (EM) clustering.

Execution:		>> python3 cluster.py [path_to_dataset, str] [class_column_header, str] K=[number_neighbors, int] "X=[input_feature_name, str]" "X=[input_feature_name, str]"
	Examples:		>> python3 cluster.py ../data/iris_preproc.csv				# K=2 by default
					>> python3 cluster.py ../data/iris_preproc.csv K=3			# To find 3 clusters in iris, for example
					>> python3 cluster.py ../data/iris_preproc.csv K=3  D=2		# To use PCA with 2 PCs, for example

Requires visualization.py and pca.py in the same folder


@author Boland Unfug
@date April 19th 2022
'''

import sys							# command line parameters
import os							# file path formatting

import numpy as np					# matrix math
import matplotlib.pyplot as plt		# plotting
from matplotlib import cm			# colormap definitions, e.g. "viridis"

import visualization as vis			# file I/O
import pca_template as pca

# TODO:
def calc_inertia( X, C, means ):
	''' Calculate the within-cluster sum of squares.  \n

	ARGS \n
	-- X: (n,m) ndarray of n training samples in m-dimensional space \n
	-- C: (n,) ndarray of n cluster assignments, one per sample in X \n
	-- means: (k,m) ndarray of k cluster's means

	RETURNS \n
	-- inertia: float, the sum of squared distances from each sample to the mean of its assigned cluster
	'''
	inertia = None
	return inertia


# TODO:
def k_means( X, k=2, animate=False, headers=[], title="" ):
	'''Perform K-Means clustering. \n
	
	ARgs: \n
		X: (n, m) ndarray of n training samples (rows), each with m input features (columns) \n
		k: int, number of clusters, \n
		animate: bool, if True iterations of the K-means algorithm will be animated at a rate of 0.5 FPS \n
		headers: list of str, if animate is True, these headers define the X and Y axis labels\n
		title: str, if animate is True, the title of the dataset can be incorporated into the figure's title\n	

	Returns: \n
		C: (n,) ndarray of cluster assignments -- one per sample in X \n
		means: (k,m) ndarray of the k clusters' means \n
		inertia: float, the within-cluster sum of squares
	'''
	C, means, inertia = None, None, None # TODO: Delete this when you're ready to debug

	# TODO: Initialize the k clusters' Means and the n samples' cluster assignments

	# TODO: Adjust the means until samples stop changing clusters

		# TODO: Check for clusters that contain zero samples, and randomly reassign their means


		# TODO: Update clusters' means and samples' distances
		
		
		# TODO: Update samples' cluster assignments


	# TODO: compute inertia
	
	return C, means, inertia


# TODO:
def e_step( X, means, covs, priors ):
	''' Update EM's responsibilities. 

	Args: \n
		X: (n,m) ndarray of n samples (rows), with m features (columns) each \n
		means: (k,m) ndarray of the k clusters' means \n
		covs: (k, m, m) ndarray of covariance matrices, one per cluster, each (m,m) \n
		priors: (k,) ndarray of k mixing coefficients (prior probability of belonging to each cluster) \n
	
	Returns: \n
		repsonsibility: (n,k) ndarray of the probabilities that each cluster explains each sample \n
	'''
	responsibility = None
	return responsibility


# TODO:
def m_step( X, responsibility ):
	''' Update EM's means, covariances, and prior probailities (mixing coefficients). 

	Args: \n
		X: (n,m) ndarray of n samples (rows), with m features (columns) each \n
		repsonsibility: (n,k) ndarray of the probabilities that each cluster explains each sample \n
	
	Returns: \n
		Nk: (k,) ndarray of responsibility-weighted cluster populations \n
		means: (k,m) ndarray of the k clusters' means \n
		covs: (k, m, m) ndarray of covariance matrices, one per cluster, each (m,m) \n
		priors: (k,) ndarray of k mixing coefficients (prior probability of belonging to each cluster) \n
	'''
	Nk, means, covs, priors = None, None, None, None	# TODO: delete

	# TODO: Update cluster populations, Nk

	# TODO: Update means and covariance matrices

	# TODO: Update mixing coefficients
	return Nk, means, covs, priors


# TODO: 
def log_likelihood( X, means, covs, priors ):
	''' Calculate the total log likelihood of samples appearing in these clusters. 

	Args: \n
		X: (n,m) ndarray of n samples (rows), with m features (columns) each \n
		means: (k,m) ndarray of the k clusters' means \n
		covs: (k, m, m) ndarray of covariance matrices, one per cluster, each (m,m) \n
		priors: (k,) ndarray of k mixing coefficients (prior probability of belonging to each cluster) \n
	
	Returns: \n
		likelihood: float, total log likelihood
	'''
	likelihood = None	# TODO: replace with a more useful equation
	return likelihood


def em( X, k=2, animate=False, headers=[], title="" ):
	'''Perform Expectation Maximization (EM) clustering. \n
	
	Args: \n
		X: (n, m) ndarray of n samples (rows), each with m features (columns) \n
		k: int, number of clusters \n

	Returns: \n
		C: (n,) ndarray of cluster assignments -- one per sample in X \n
		Nk: (k,) ndarray of responsibility-weighted cluster populations \n
		means: (k,m) ndarray of the k clusters' means \n
		covs: (k, m, m) ndarray of covariance matrices, one per cluster, each (m,m) \n
		priors: (k,) ndarray of k mixing coefficients (prior probability of belonging to each cluster) \n
		inertia: float, the within-cluster sum of squares \n
	'''	
	# Initialize
	n, m = X.shape
	priors = np.ones((k,)) / k							# cluster mixing coefficients
	means = np.zeros((k,m))
	covs = np.zeros((k,m,m))
	for cluster in range(k):
		idx = np.random.randint(0, n)
		means[cluster,:] = X[idx,:]						# cluster means
		covs[cluster,:,:] = np.cov( X, rowvar=False )/k	# cluster covariance matrices

	iteration_threshold = 500
	convergence_threshold = 0.001
	likelihood = np.zeros((iteration_threshold,))

	# Compute the first iteration in order to evaluate the while loop's condition
	likelihood[0] = log_likelihood( X, means, covs, priors )
	responsibility = e_step( X, means, covs, priors )
	Nk, means, covs, priors = m_step( X, responsibility )
	iteration = 1
	likelihood[iteration] = log_likelihood( X, means, covs, priors )

	if animate:
		inertia = np.zeros((iteration_threshold,))
		if k < 6:
			fig, axs = plt.subplots(1,k)
		else:
			fig, axs = plt.subplots(2,k//2)
			axs = axs.flatten()
		plt.suptitle( f"{title}\niteration {iteration:d}, log likelihood {likelihood[iteration]:.2f}")
		for ax in axs:
			ax.grid( True )
			ax.set_xlabel( headers[0] )
			ax.set_ylabel( headers[1] )
			ax.set_title( f"Cluster {cluster:d}")
		plt.tight_layout()
		C = np.argmax( responsibility, axis=1 )
		inertia[iteration] = calc_inertia( X, C, means )
		dots = [0] * k
		diamonds = [0] * k * 2
		plt.suptitle( f"{title}\niteration {iteration:d}, log likelihood {likelihood[iteration]:.2f}")
		for cluster in range(k):
			axs[cluster].scatter( X[:,0], X[:,1], c=responsibility[:,cluster] )
			axs[cluster].plot( means[:,0],  means[:,1], 'dm', markeredgecolor='w', markersize=20, linewidth=2, alpha=0.40 )
			axs[cluster].plot( means[cluster,0],  means[cluster,1], 'dm', markeredgecolor='w', markersize=20, linewidth=2 )
		plt.pause( 0.25 )

	# Iterate until log likelihood converges. (It doesn't matter what the log likelihood's value is, just that it stops changing.)
	done = (k==0) or (np.abs(likelihood[iteration] - likelihood[iteration-1]) <= convergence_threshold) or (iteration_threshold <= iteration)
	while not done:
		iteration += 1

		# Remove empty clusters
		populated = np.logical_not( np.logical_or((Nk == 0.0), np.isnan( Nk )) )
		if np.sum( populated ) < k:
			print( f"\nREMOVING EMPTY CLUSTER {np.argmin( populated )}" )
			Nk = Nk[ populated ]
			priors = priors[ populated ]
			means = means[ populated, : ]
			covs = covs[ populated, :, : ]
			responsibility = responsibility[ :, populated ]
			k = len( Nk )

		# EM iteration
		responsibility = e_step( X, means, covs, priors )
		Nk, means, covs, priors = m_step( X, responsibility )
		likelihood[iteration] = log_likelihood( X, means, covs, priors )
		done = (k==0) or (np.abs(likelihood[iteration] - likelihood[iteration-1]) <= convergence_threshold) or (iteration_threshold <= iteration)

		if np.sum( np.isnan( covs )) > 0.0:
			for cluster in range(k):
				print( f"\niteration {iteration}" )
				print( f"Nk[{cluster:d}] = {Nk[cluster]:.1f}" )
				print( f"priors[{cluster:d}] = {priors[cluster]:.3f}" )
				print( f"mean( resp[:,{cluster:d}] ) = {np.mean( responsibility[:,cluster] ):.3f}" )
				print( f"means[{cluster:d},:] =\n", means[ cluster, : ] )
				print( f"covs[{cluster:d},:,:] =\n", covs[ cluster, :, : ] )

		# Visualize EM's progress by drawing every 20-th update step
		if animate:
			C = np.argmax( responsibility, axis=1 )
			inertia[iteration] = calc_inertia( X, C, means )
			plt.suptitle( f"{title}\niteration {iteration:d}, log likelihood {likelihood[iteration]:.2f}")
			for cluster in range(k):
				axs[cluster].clear()
				axs[cluster].scatter( X[:,0], X[:,1], c=responsibility[:,cluster] )
				axs[cluster].plot( means[:,0],  means[:,1], 'dm', markeredgecolor='w', markersize=20, linewidth=2, alpha=0.40 )
				axs[cluster].plot( means[cluster,0],  means[cluster,1], 'dm', markeredgecolor='w', markersize=20, linewidth=2 )
			plt.pause( 0.25 )

	if animate:
		plt.pause( 3 )
		# Visualize cluster performance metrics
		fig, axs = plt.subplots(2,1)
		plt.suptitle( f"{title}: Performance" )
		axs[0].set_ylabel( "Log Likelihood")
		axs[0].set_xlabel( "Iteration")
		axs[0].plot( likelihood[:iteration], '-' )
		axs[1].set_ylabel( "Inertia")
		axs[1].set_xlabel( "Iteration")
		inertia[0:2] = inertia[2]
		axs[1].plot( inertia[:iteration], '-' )
		plt.tight_layout()
		plt.pause( 3 )

	# Final cluster assignments and overall inertia
	C = np.argmax( responsibility, axis=1 )
	return C, Nk, means, covs, priors, inertia


def print_cluster_stats( X, C, title="" ):
	'''Report the overall inertia as well as each cluster's mean, covariance matrix, and population (number of samples). \n
	
	ARgs: \n
		X: (n, m) ndarray of n training samples (rows), each with m input features (columns) \n
		C: (n,) ndarray of cluster assignments -- one per sample in X \n
		title: str, text to display at the top of the printout, e.g. f"{dataset_name}: {algorithm_name} Clustering"
	'''
	# Determine number of clusters, and their names
	cluster_id, population = np.unique( C, return_counts=True )
	k = len( cluster_id ) 

	# Measure overall inertia
	n, m = X.shape
	means = np.zeros((k,m))
	for cluster in range( k ):
		cid = cluster_id[ cluster ]
		in_cluster = C == cid 
		X_c = X[ in_cluster, : ]
		means[ cid, : ] = np.mean(X_c, axis=0)
	inertia = calc_inertia( X, C, means )

	# Display cluster characteristics in the terminal
	print( f"\n\n{title.upper()}" )
	print( f"clusters = {k:d}" )
	print( f"inertia  = {inertia:.3f}" )
	for cluster in range( k ):
		cid = cluster_id[ cluster ]
		in_cluster = C == cid 
		X_c = X[ in_cluster, : ]
		print( f"\nCLUSTER {cid:d}:"  )
		print( f"population  =  {population[cid]} samples  ({population[cid]/n*100:.1f}%)" )
		print( f"mean  =  {means[ cid, : ]}" )
		print( f"cov   =\n{np.cov(X_c, rowvar=False)}" )
	print( "\n" )


def gaussian( x, mean, cov ):
	''' Probability of the sample x appearing in a normal distribution with this mean and covariance.\n

	Args:\n
		x: (m,) ndarray, one sample (row) from the dataset\n
		mean: (m,) ndarray, the mean (row) of one cluster\n
		cov: (m,m) ndarray, the covariance matrix of one cluster\n 
	
	Returns:\n
		prob: float, p( x | mean, cov )
	'''
	m = cov.shape[0]
	cov_det = np.linalg.det( cov )
	cov_inv = np.linalg.pinv( cov )
	if cov_det < 0.0001:
		# Prevent covariance matrix singularities from crashing EM
		cov_det = 1.0
		cov_inv = np.eye( m )
	coeff = 1/( 2*np.pi**m * cov_det )**0.5
	exponent = -0.5 * (x - mean) @ cov_inv @ (x - mean).T
	prob = coeff * np.exp( exponent )
	return prob


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
		filepath = os.path.join(current_directory, "data", "iris_preproc.csv")

	# Read in the dataset and remove NaNs
	data, headers, title = vis.read_csv( filepath )
	data, headers = vis.remove_nans( data, headers )
	
	# Let the user name a class label feature (the target to predict), input feature(s), and/or the number of nearest
	# neighbors, k. Use commandline parameter prefixes: "K=" for number of nearest neighbors, "C=" for class feature, 
	# "X=" for each input feature. By default it will set K=1 and use the rightmost column as the class label feature.
	k = 1
	n = data.shape[0]
	d = 0
	X_headers = []
	X = None
	if len(argv) > 2:
		for param in argv[2:]:
			
			if "K=" in param:	
				# Let the user specify the maximum polynomial degree that they want to use in the model with the syntax "D=2", where 2 could be any integer
				k_param = param.split("=")[1]
				k = int( k_param )
				print( f"K-Means and EM configured to fit K = {k} clusters" )
			
			elif "D=" in param:	
				# Let the user specify the maximum polynomial degree that they want to use in the model with the syntax "D=2", where 2 could be any integer
				d_param = param.split("=")[1]
				d = int( d_param )
				print( f"PCA configured to project onto {d} dimensions" )

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
					print( f"X input feature(s) selected: {X_headers}" )
				else:
					print( f"\nWARNING: '{X_param}' not found in the headers list: {headers}. Not selected as an X input feauture selected.\n" )	

	# If the user has not specified any X features, use every feature except C as input, 1 at a time
	if len(X_headers) < 1:
		X = data[:,:]
		X_headers = headers[:]

	# Cluster in original feature space
	C_km, means_km, km_inertia = k_means( X, k, True, X_headers, title=f"{title} K-Means (k={k:d})" )
	print_cluster_stats( X, C_km, title=f"{title} K-Means (k={k:d})" )
	C_em, Nk_em, means_em, covs_em, priors_em, inertia_em = em( X, k, True, X_headers, title=f"{title} EM (k={k:d})" )
	print_cluster_stats( X, C_em, title=f"{title} EM (k={k:d})" )

	if d > 0:
		# PCA
		n, m = X.shape
		X_norm, _, _ = pca.z_transform( X )
		if n > m:
			P, e_scaled = pca.pca_cov( X_norm )
		else:
			P, e_scaled = pca.pca_svd( X_norm )
		pca.pc_heatmap( P, e_scaled, X_headers )
		Y = pca.rotate( X_norm, P )
		Y = pca.project( Y, d )
		
		# Y's headers are the PC names and their scaled eigenvalues
		Y_headers = []
		for p in range(d):
			Y_headers.append( f"PC{p}\ne{p}={e_scaled[p]:.2f}" )

		# Cluster in PC-space
		C_km, means_km, km_inertia = k_means( Y, k, True, Y_headers, title=f"{title} PCA (d={d:d}) + K-Means (k={k:d})" )
		print_cluster_stats( Y, C_km, title=f"{title} PCA (d={d:d}) + K-Means (k={k:d})" )
		C_em, Nk_em, means_em, covs_em, priors_em, inertia_em = em( Y, k, True, Y_headers, title=f"{title} PCA (d={d:d}) + EM (k={k:d})" )
		print_cluster_stats( Y, C_em, title=f"{title} PCA (d={d:d}) + EM (k={k:d})" )


if __name__=="__main__":
	main( sys.argv )
	plt.show()