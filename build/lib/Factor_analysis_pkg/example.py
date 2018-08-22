import numpy as np
from numpy import linalg as la
from sklearn import datasets

from .EM_Factor_analysis import factor_analysis

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# iris data example

def run_demo():

	print("play with iris data")
	
	iris = datasets.load_iris()
	x = iris.data
	y = iris.target

	print("Assume 2 latent variables")
	nFactors = 2 # assume 2 latent variables

	[W, z, log_history] = factor_analysis(x, nFactors)

	plt.scatter(x = z[:,0], y = z[:,1],c = y)
	plt.title("iris dataset reduced to 2 Factors")
	plt.show()

	plt.plot(log_history)
	plt.title("log likelihood over iteration")
	plt.show()

	print("Assume 3 latent variables")
	nFactors = 3 # assume 3 latent variables this time

	[W, z, log_history] = factor_analysis(x, nFactors)


	fig = plt.figure()
	ax = fig.add_subplot(111, projection = '3d')
	ax.scatter(xs = z[:,0], ys = z[:,1], zs = z[:,2], c = y)
	plt.title("iris dataset reduced to 3 Factors")
	plt.show()

	plt.plot(log_history)
	plt.title("log likelihood over iteration")
	plt.show()