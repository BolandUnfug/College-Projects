# lfw_crop_challenge_template.py -- Demo opening lfwcrop.npy and plotting the first face, a random face, and
# 	the mean face. Use lfwcrop_ids.txt to label individuals. Maybe we'll get to PCA-SVD, but probably not.
# Caitrin Eaton
# Machine Learning for Visual Thinkers
# Fall 2020

import os
from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani


def read_lfwcrop():
	''' Return an ndarray of LFW Crop's image data and a list of the corresponding names. '''

	# Read the images in from the lfwcrop.npy file
	faces_filename = "lfwcrop.npy"
	current_directory = os.path.dirname(__file__)
	faces_filepath = os.path.join(current_directory, "data", faces_filename)
	lfw_faces = np.load( faces_filepath )

	# Read the name of each image in from the lfwcrop_ids.txt file
	names_filename = "lfwcrop_ids.txt"
	names_filepath = os.path.join(current_directory, "data", names_filename)
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
	# plt.figure()
	# plt.imshow( first_face, cmap="bone" )
	# plt.title( first_name )

	# Visualize a random face
	random_face_num = np.random.randint(0, n)
	random_face = X[random_face_num,:,:]
	random_name = lfw_names[random_face_num]
	plt.figure()
	plt.imshow( random_face, cmap="bone" )
	plt.title( random_name )

	# Visualize the mean face
	mean_face = np.mean(X, axis=0)
	mean_name = "Mean Face"
	# plt.figure()
	# plt.imshow( mean_face, cmap="bone" )
	# plt.title( mean_name )

	# PCA-Cov?
	# X has more than 3 dimensions, so covariance won't work. Needs to be reshaped
	# PCA-COV breaks when there are more dimensions than samples, so we will do PCA-SVD instead
	
	n = 100
	X = X[random_face_num:n+random_face_num, :, :]
	X = X.reshape((n, m))
	X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
	print(f"X_norm.shape = {X_norm.shape}")

	# PCA-COV
	# C = np.cov(X_norm, rowvar=False)
	# e, P = np.linalg.eig(C)

	(U, W, Vt) = np.linalg.svd(X_norm)
	e = W**2 / np.sum(W**2)
	P = Vt.T
	print(f"P.shape = {P.shape}")

	# Set PCs in order of descending eigenvalues
	order = np.argsort(e)[::-1]
	e = e[order]
	P = P[:, order]
	print(f"Eigenvalues = {e}")

	# Visualize PC heatmap
	# plt.figure()
	# plt.imshow(P, cmap="viridis")
	# plt.title("PCs")

	# Scree plot
	# plt.figure()
	# plt.plot(e, "-")
	# plt.title("Eigenvalues")

	# First PC?
	P0 = P[:, 0].reshape((64, 64))
	# plt.figure()
	# plt.imshow( P0, cmap="bone" )
	# plt.title( "P0" )

	# Face space
	Y = X_norm @ P
	# plt.figure()
	# plt.plot(Y[:,0], Y[:, 1], "o")
	# plt.xlabel("P0")
	# plt.ylabel("P1")
	# plt.title("Face Space")
	
	
	
	fig,ax = plt.subplots()
	rec_name = random_name  # updates title
	plt.title( rec_name ) # updates title
	# Project and reconstruct
	def update(frame):	
		d = frame
		Y_proj = Y[:, 0:d] # projects to dimension
		X_rec = (Y_proj @ P[:, 0:d].T) * np.std(X, axis=0) + np.mean(X, axis=0) # changes the reconstruction
		rec_face = X_rec[0, :].reshape((64, 64)) # applies changed reconstruction to face
		face = ax.imshow( rec_face, cmap="bone") # updates graph
		return face,
	
	
	anim = ani.FuncAnimation(fig,update,frames=n,interval=10000/n, blit=True)
	
	plt.show()



if __name__ == "__main__":
	main()
