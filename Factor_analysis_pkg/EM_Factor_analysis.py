import numpy as np
from numpy import linalg as la

def factor_analysis(x, nFactors):
	'''
	EM Factor analysis code
	Author: Donnie Kim
	INPUT:
		x: float numpy array of p by k matrix where p is the number of data points and k is the number of observed variables
		nFactors: Integer. Number of hidden latent variables to be considered
	output: 
		lambda: Float numpy array. Factor loading matrix in dimension of p by k
		psi: Float numpy array. k by k matrix
		mu: Float numpy array. a vector of mean value in dimension of 1 by k
	
	Referenced material:
	1.http://www.cs.toronto.edu/~fritz/absps/tr-96-1.pdf
	2.Pattern Recognition and Machine Learning by Bishop
	3.Bayseian Reasoning and Machine Learning by Barber

	nData == p
	nDim == k
	lambda == W
	'''

	nData, nDim = x.shape # obtain number of data points and number of observed var
	mu = x.mean(axis=0) # obtain the mean value in each observed var
	x_scaled = x - mu # x has a zero mean and cov of lambda*lambda^T + psi
	
	Covariance = np.cov(x_scaled.T) # obtain covariance of x

	Psi = np.diag(Covariance.diagonal()) # diagonal matrix

	Scaling = la.det(Covariance)**(1./nData) # used to ensure that when intergrated, it sums up to 1
 
 	# Used later in E and M steps. Pre-define them here.
	I = np.eye(nFactors)
	W = np.random.normal(0,np.sqrt(Scaling/nFactors),(nDim,nFactors))

	old_LL = -np.inf # initialize log-likelihood to -infinity
	log_history = [old_LL]

	nIterations=1000	
	for i in range(nIterations):
		   
		# E-step
		WPsi = np.dot(W.T, la.inv(Psi))
		G = la.inv(I+np.dot(WPsi,W)) # 12.68 from Bishop

		Ez = np.dot(G,np.dot(WPsi,x_scaled.T)) # 12.66 from Bishop
		Ezz = G + np.dot(Ez,Ez.T) # 12.67 from Bishop

		# M step
		x_scaledEz = np.dot(x_scaled.T,Ez.T)

		W = np.dot(x_scaledEz,la.inv(Ezz)) # 12.69 from Bishop
		Psi = np.diag(np.diag(Covariance-np.dot(W,x_scaledEz.T)/nData)) # 12.70 from Bishop

		# Log-likelihood
		Sigma = np.dot(W,np.transpose(W)) + Psi # 21.1.11 from Barber
		logdet = np.log(la.det(2*np.pi*Sigma))
		
		LL = -nData/2*(np.trace(np.dot(la.inv(Sigma),Covariance.T)) + logdet) # 21.1.13 from Barber

		# record the log-likelihood
		log_history.append(LL)

		if (LL-old_LL)<(1e-4):
			break
		old_LL = LL

	# Obtain  z (or Ez)
	InvSigma = la.inv(np.dot(W, np.transpose(W)) + Psi)
	z = np.dot(x_scaled, np.dot(np.transpose(InvSigma), W) )

	return W, z, log_history