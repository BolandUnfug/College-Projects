# MLVT STUDENTS: CTRL+F for "# TODO:" to jump to sections of code that have been left for you to complete
'''
statistics.py

Statistical analysis of data read into a Numpy ndarray from a CSV file. Annotates pair plots and
scatter plots with means, medians, and standard deviations. Visualizes the covariance matrix as
a heatmap.

Execution:		>> python3 statistics.py [path_to_dataset, str] [class_column_header, str]
	Examples: 		>> python3 statistics.py 
					>> python3 statistics.py ../data/iris.csv
					>> python3 statistics.py ../data/iris.csv species

Requires visualization.py in the same folder

# TODO:
@author Boland Unfug
@date 2/8/2022
'''


from visualization import *		# includes numpy, matplotlib, os, and sys
import math

# For drawing elliptical patches to represent clusters' standard deviations
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.colors as colors


# MLVT students: You DO need to edit this function. The comments will help!
def scatter_gauss( data, headers, col_x, col_y, col_c=None, title="Gaussian Distribution" ):
	''' Annotate scatter plot with means, medians, and standard deviations.

	ARGS:
	data (ndarray), dataset that contains columns with the X and Y data to be plotted
	header (list of str), the names of each feature in the data ndarray
	col_x (int), column index of the X feature within the data ndarray
	col_y (int), column index of the Y feature within the data ndarray
	col_c (int), OPTIONAL: column index of the feature to use as a marker color code
	title (str), OPTIONAL: text that will appear immediately above the plot

	RETURN:
	fig (figure reference), the figure in which the new plot appears
	'''

	# Start with a scatter plot, then suprimpose gaussian ellispe(s) on top
	fig = scatter( data, headers, col_x, col_y, col_c, title )
	ax = plt.gca()

	# Plot samples, color-coding if necessary
	data_xy = data[ :, [col_x, col_y] ]
	colorcoded = (col_c != None) and (col_c < data.shape[1])
	if colorcoded:
		print("trigered multi data")
		# Compute a separate distribution within each distinct class
		class_labels = np.unique( data[ :, col_c ] )	# determine if there is more than one class label present
		num_classes = len( class_labels )
		
		# Plot each class (i.e. each subset of matching values in col_c) in its own color using a
		# colorblind-friendly colormap: "viridis", "plasma", "inferno", "magma", or "cividis"
		cmap = cm.get_cmap( "viridis", num_classes ) 

		for i in range( num_classes ):

			# Find the samples that belong to this specific class
			label = class_labels[ i ]
			print(label)
			has_lbl = data[ :, col_c ] == label
			color = cmap( i/(num_classes-1) )

			
			# Compute the distribution within this specific class
			# same as other thing, but you specify the row with i? or is it column?
			mean = np.mean(data_xy[has_lbl , :], axis=0)
			cov = np.cov(data_xy[has_lbl , :], rowvar=False)

			ax.plot(mean[0],mean[1],'dk', markerfacecolor=color, markersize=15, alpha=0.5, label=f"{label} mean" )
			for n_std in range(1,4):
				gaussian_ellipse( mean, cov, ax, n_std, color=color )
			print( "\nTO DO: in scatter_gauss(), superimpose CLASS means and covariance ellipses on scatter plots")
			'''
			# 1. compute the mean within this class
			# 2. compute the covariance matrix within this class
			# 3. plot the mean as a large diamond, e.g. print ax.plot( ?, ?, 'dk', markerfacecolor=color, markersize=15, alpha=0.5, label=f"{label} mean" )
			# 4. Visualize distribution shape by drawing ellispses at the 1st, 2nd, and 3rd standard deviations
			
				
			'''
	else:
		# TODO:
		# No class feature specified, so compute the distribution of the entire dataset
		print("Triggered Else statement")
		mean = np.mean(data_xy, axis=0)
		cov = np.cov(data_xy, rowvar=False)
		for n_std in range(1,4):
			gaussian_ellipse( mean, cov, ax, n_std)
		print( "\nTODO: in scatter_gauss(), superimpose OVERALL mean and covariance ellipses on scatter plots")
		'''
		# 1. compute the mean
		# 2. compute the covariance matrix
		# 3. plot the mean as a large diamond, e.g. print ax.plot( ?, ?, 'dk', markerfacecolor=color, markersize=15, alpha=0.5, label=f"{label} mean" )
		# 4. Visualize distribution shape by drawing ellispses at the 1st, 2nd, and 3rd standard deviations
		for n_std in range(1,4):
			gaussian_ellipse( mean, cov, ax, n_std, color=color )
		'''
		
	ax.legend()

	return fig


# MLVT students: You DO need to edit this function. The comments will help!
def report_stats( data, headers, col_c=None ):
	''' Print each feature's means, medians, standard deviations, variances, and covariance matrix.

	ARGS:
	data (ndarray), dataset that contains columns with the X and Y data to be plotted
	header (list of str), the names of each feature in the data ndarray
	col_c (int), OPTIONAL: column index of the feature to use as a marker color code

	RETURN:
	None
	'''
	
	# Format the number of decimal places that appear in numpy arrays during printing
	np.set_printoptions( precision=6, floatmode="fixed" )

	# TODO:
	# Print statistics of the entire dataset, as a whole, regardless of class labels
	min = np.min(data, axis=0)
	max = np.max(data, axis=0)
	rnge = np.ptp(data, axis=0)
	median = np.median(data,axis=0)
	mean = np.mean(data, axis=0)
	std = np.std(data)
	var = np.var(data)
	cov = np.cov(data, rowvar=False)
	print( f"\nSTATS: ENTIRE DATASET" )
	print( f"Features:\t{headers}" )
	print( f"Min:     \t{min}" )
	print( f"Max:     \t{max}" )
	print( f"Range:   \t{rnge}" )
	print( f"Median:  \t{median}" )
	print( f"Mean:    \t{mean}" )
	print( f"Std Dev: \t{std}" )
	print( f"Variance:\t{var}" )
	print( f"Cov:     \n{cov}\n" )

	print("\nTODO: Print the stats of the entire dataset, regardless of class labels")
	'''
	min    = ?
	max    = ?
	rnge   = ?
	median = ?
	mean   = ?
	std    = ?
	var    = ?
	cov    = ?
	
	'''

	# If there is a column of class labels, use this info for a class-specific stats breakdown
	if (col_c != None) and (col_c < data.shape[1]):
		print("entered loop")
		# Reporting stats on the class label itself makes no sense; it isn't really a numeric feature
		cols_feat = list(range( data.shape[1] ))
		cols_feat.pop( col_c )
		headers_feat = headers[:]
		headers_feat.pop( col_c )
		data_feat = data[ :, cols_feat ]

		class_labels = np.unique( data[ :, col_c ] )
		num_classes = len( class_labels )
		for i in range( num_classes ):

			# Find the samples that belong to this specific class
			label = class_labels[ i ]
			has_lbl = data[ :, col_c ] == label

			# TODO:
			# Print the stats within this specific class
			print("\nTODO: Print the stats within each individual class")
			'''
			min    = ?
			max    = ?
			rnge   = ?
			median = ?
			mean   = ?
			std    = ?
			var    = ?
			cov    = ?

			print( f"\nSTATS: WITHIN CLASS {label}" )
			print( f"Features:\t{headers_feat}" )
			print( f"Min:     \t{min}" )
			print( f"Max:     \t{max}" )
			print( f"Range:   \t{rnge}" )
			print( f"Median:  \t{median}" )
			print( f"Mean:    \t{mean}" )
			print( f"Std Dev: \t{std}" )
			print( f"Variance:\t{var}" )
			print( f"Cov:     \n{cov}\n" )
			'''


# MLVT students: You do NOT need to edit this function.
def gaussian_ellipse(mean, cov, ax, n_std=3.0, color='b', linewidth=2.0):
	"""
	Outline a gaussian's standard deviations, given its mean and covariance matrix.
	Adapted from https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html.

	ARGS:
	mean (ndarray) with shape (2,) defines the center of the gaussian ellipse
	cov (ndarray) with shape (2,2) defines the orientation of the gaussian ellipse
	ax (matplotlib.axes.Axes) the axes object containing the scatter plot onto which the ellipse will be superimposed
	n_std (float) the number of standard deviations to determine the ellipse's radii
	kwargs [OPTIONAL] matplotlib.patches.Patch properties

	RETURNS:
	matplotlib.patches.Ellipse
	"""
	pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1]) # throwing an error
	# Using a special case to obtain the eigenvalues of this two-dimensional dataset.
	ell_radius_x = np.sqrt(1 + pearson)
	ell_radius_y = np.sqrt(1 - pearson)
	ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
					  facecolor=color, alpha=0.15)

	# Calculating the stdandard deviation of x from the squareroot of the variance and multiplying 
	# with the given number of standard deviations.
	scale_x = np.sqrt(cov[0, 0]) * n_std
	mean_x = mean[0]

	# calculating the stdandard deviation of y from the squareroot of the variance and multiplying 
	# with the given number of standard deviations.
	scale_y = np.sqrt(cov[1, 1]) * n_std
	mean_y = mean[1]

	transf = transforms.Affine2D() \
		.rotate_deg(45) \
		.scale(scale_x, scale_y) \
		.translate(mean_x, mean_y)

	ellipse.set_transform(transf + ax.transData)
	return ax.add_patch(ellipse)


# MLVT students: You do NOT need to edit this function.
def heatmap_cov( data, headers, col_c=None, title="Covariance" ):
	''' Annotate scatter plot with means, medians, and standard deviations.

	ARGS:
	data (ndarray), dataset with shape (m,n): m features & n samples
	header (list of str), the names of each feature in the data ndarray
	col_c (int), OPTIONAL: column index of the feature to use as a marker color code
	title (str), OPTIONAL: text that will appear immediately above the plot

	RETURN:
	None
	'''

	# Overall covariance regardless of class label
	cov = np.cov( data, rowvar=False )
	fig = heatmap( cov, headers, title)

	# Replace Y-axis tick marks (usually the sample number, in a heatmap) with feature names
	ax = plt.gca()
	ax.set_ylabel( "features" )	
	ax.set_yticks( np.arange(len(headers)) )
	ax.set_yticklabels( headers )
	plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor") # Rotate the column labels and set their alignment.
	fig.tight_layout()

	# If there is a column of class labels, use this info for a class-specific stats breakdown
	if (col_c != None) and (col_c < data.shape[1]):
		
		class_labels = np.unique( data[ :, col_c ] )
		num_classes = len( class_labels )

		# Heatmapping the class label itself makes no sense; it isn't really a numeric feature
		cols_feat = list(range( data.shape[1] ))
		cols_feat.pop( col_c )
		headers_feat = headers[:]
		headers_feat.pop( col_c )
		data_feat = data[ :, cols_feat ]

		for i in range( num_classes ):

			# Find the samples that belong to this specific class
			label = class_labels[ i ]
			has_lbl = data[ :, col_c ] == label

			# Visualize covariance within this specific class
			class_title = f"{title} within class {label}"
			cov = np.cov( data_feat[ has_lbl, : ], rowvar=False )
			heatmap( cov, headers_feat, class_title )

			# Replace Y-axis tick marks (usually the sample number, in a heatmap) with feature names
			ax = plt.gca()
			ax.set_ylabel( "features" )	
			ax.set_yticks( np.arange(len(headers_feat)) )
			ax.set_yticklabels( headers_feat )
			plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor") # Rotate the column labels and set their alignment.
			fig.tight_layout()


# MLVT students: You DO need to edit this function. :)
def main( argv ):
	''' Parse command line arguments: 
		-- argv[0] is always the name of the program run from the terminal
		-- argv[1] should be the path of a data file (e.g. *.DATA or *.CSV) to read into a Pandas DataFrame
		-- argv[2] (optional) can be used to specify the name of a feature to use as a class color code
	'''

	# Determne the input file's path: either it was supplied in the commandline, or we should use iris as a default
	if len(argv) > 1:
		filepath = argv[1].strip()
	else:
		#filepath = "../data/iris_preproc.csv"
		current_directory = os.path.dirname(__file__)
		filepath = os.path.join(current_directory, "data", "iris_preproc.csv")

	# Read the dataset into a numpy ndarray object (a matrix)
	data, headers, title = read_csv( filepath )

	# Remove non-numeric ("NaN") features and rows with missing values, or else stats will be NaN as well
	data, headers = remove_nans( data, headers )
	
	# Let the user name a feature that they want to use as the class color code in the plots below
	class_col = None
	if len(argv) > 2:
		try: 
			# Find the column that contains the user's chosen feature name 
			class_col = headers.index( argv[2] )	# CAUTION: this will crash if the user's input does not match any element in the list of headers
		except:
			print( f"\nWARNING: '{argv[2]}' not found in the headers list: {headers}. No class color coding applied.\n" )

	# Compute the stats!
	report_stats( data, headers, col_c=class_col )

	# Visualize the stats!
	# turn into a system to graph each relationship against one
	heatmap_cov( data, headers, col_c=class_col, title=f"{title} Covariance" )
	scatter_gauss( data, headers, col_x=1, col_y=12, col_c=class_col, title=title )
	scatter_gauss( data, headers, col_x=2, col_y=12, col_c=class_col, title=title )
	scatter_gauss( data, headers, col_x=3, col_y=12, col_c=class_col, title=title )
	scatter_gauss( data, headers, col_x=4, col_y=12, col_c=class_col, title=title )
	scatter_gauss( data, headers, col_x=5, col_y=12, col_c=class_col, title=title )
	scatter_gauss( data, headers, col_x=6, col_y=12, col_c=class_col, title=title )
	scatter_gauss( data, headers, col_x=7, col_y=12, col_c=class_col, title=title )
	scatter_gauss( data, headers, col_x=8, col_y=12, col_c=class_col, title=title )
	scatter_gauss( data, headers, col_x=9, col_y=12, col_c=class_col, title=title )
	scatter_gauss( data, headers, col_x=10, col_y=12, col_c=class_col, title=title )
	scatter_gauss( data, headers, col_x=11, col_y=12, col_c=class_col, title=title )




	# TODO:
	print( "\nTO DO: Repeat scatter_gauss() for as many pairs of features as you'd like. Maybe even in a nested loop (optional).")

	plt.show()  # throwing an error


if __name__=="__main__":
	main( sys.argv )
