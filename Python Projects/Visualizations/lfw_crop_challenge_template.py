# lfw_crop_challenge_template.py -- Demo opening lfwcrop.npy and plotting the first face, a random face, and
# 	the mean face. Use lfwcrop_ids.txt to label individuals. Maybe we'll get to PCA-SVD, but probably not.
# Caitrin Eaton
# Machine Learning for Visual Thinkers
# Fall 2020

import os
import numpy as np
import matplotlib.pyplot as plt
import random as rand
import pca_template as pca

def read_lfwcrop():
	''' Return an ndarray of LFW Crop's image data and a list of the corresponding names. '''

	# Read the images in from the lfwcrop.npy file
	faces_filename = "lfwcrop.npy"
	current_directory = os.path.dirname(__file__)
	faces_filepath = os.path.join(current_directory, "data", faces_filename)
	lfw_faces = np.load( faces_filepath )

	# Read the name of each image in from the lfwcrop_ids.txt file
	names_filename = "lfwcrop_ids.txt"
	names_filepath = os.path.join(current_directory,  "data", names_filename)
	lfw_names = np.loadtxt( names_filepath, dtype=str, delimiter="\n" )
	
	return lfw_faces, lfw_names


def plot_face( iamge, title, ax=None ):
	'''Given an image in a 2D ndarray and the desired figure title, visualizes the image as a heatmap.
	Optional ax parameter: can provide a particular axis object in which to display the image. Returns 
	nothing (None).'''
	pass


def main():
	''' Draw some faces from the LFW Crop dataset. Draw the mean face. Try out PCA-Cov and PCA-SVD.
	Try out a reconstruction.'''

	# Read in the dataset, including all images and the names of the associated people
	X, lfw_names = read_lfwcrop()
	n = X.shape[0]
	m = X.shape[1]*X.shape[2]
	print( "faces:", X.shape )
	print( "names:", len(lfw_names) )
	print( "features:", m )
	
	
	# Visualize the first face
	first_face = X[0,:,:]
	first_name = lfw_names[0]
	plt.figure()
	plt.imshow( first_face, cmap="bone" )
	plt.title( first_name )
	
	# Visualize a random face
	randface = rand.randrange(0,13231)
	first_face = X[randface,:,:]
	first_name = lfw_names[randface]
	plt.figure()
	plt.imshow( first_face, cmap="bone" )
	plt.title( first_name )

	# Visualize the mean face
	
	

	# PCA-Cov?


	# PCA-SVD?


	# Project and reconstruct?
	

	plt.show()


if __name__ == "__main__":
	main()