from numpy import *
import numpy as np
from scipy.linalg import eig

#=====================
# Whitens Eigenvectors
#=====================

def whiten(x, opts):
	xcent = x - mean(x,0)
	c = np.dot(transpose(xcent),xcent)
	
	D, V = eig(c)
	iD = diag(np.sqrt(1.0/D))
	trans = np.dot(np.dot(V, iD),V.transpose())
	
	xwhite = np.dot(xcent, trans)
	return xwhite
