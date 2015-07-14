import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import *
from numpy import *
import collections
import timing

#===========================================================================
# 'Normalized Cuts', finds k non-zero eigenvectors of matrix and normalizes.
#===========================================================================
def ncuts(a_down, opts, k=16):
	a, b = a_down.shape
	sm = a_down.sum(0)
	sm = np.squeeze(np.asarray(sm))

	d = sp.csc_matrix((sm, (np.arange(a), np.arange(a))), shape=(a,b))
	nvec = k + 1
	
	if opts.verbose: print 'Sparse Matrix created'
	e = (d - a_down)
	e = e + sp.identity(a, format='csc')*(10.0**-10)
	
	if opts.verbose: timing.log('Solving for eigens...')
	Eval, Ev = eigs(e, k=nvec, M=d, sigma=0.0)
	if opts.verbose: timing.log('Solved for eigens!')

	#Sort vectors in descending order, leaving out the zero vector
	idx = np.argsort(-Eval)
	Ev = Ev[:,idx[-2::-1]].real
	Eval = Eval[idx[-2::-1]].real
	if opts.debug: print 'Ev: ', Ev
	
	#Make vectors unit norm
	tmp = Ev**2
	tmp = tmp.sum(0)
	tmp = np.sqrt(tmp)
	Ev = Ev/tmp

	eigens = collections.namedtuple('eigens', ['ev','evl'])
	eig = eigens(ev=Ev, evl=Eval)
	return eig
