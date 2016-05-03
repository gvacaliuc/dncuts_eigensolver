import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import *
from numpy import *
import collections
import timing
import argparse

#===========================================================================
# 'Normalized Cuts', finds k non-zero eigenvectors of matrix and normalizes.
#===========================================================================
def ncuts(a_down, opts, k=16):
	a, b = a_down.shape
	sm = np.array(a_down.sum(0)).flatten();

	d = sp.diags(sm,0,format='csc');
	nvec = k + 1
	
	if opts.verbose: timing.log('Solving for eigens...')
	Eval, Ev = eigsh(a_down, k=nvec, M=d, sigma=0.0)
	if opts.verbose: timing.log('Solved for eigens!')

	#Sort vectors in descending order, leaving out the zero vector
	idx = np.argsort(-Eval)
	Ev = Ev[:,idx[1:]].real
	Eval = Eval[idx[1:]].real
	if opts.debug: print 'Ev: ', Ev
	
	#Make vectors unit norm
	for i in range(k):
		Ev[i] /= np.linalg.norm(Ev[i]);

	eigens = collections.namedtuple('eigens', ['ev','evl'])
	eig = eigens(ev=Ev, evl=Eval)
	return eig;

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--array', type=str, dest='aff', default='Null', help = 'Path to .npz file containing CSC sparse array');
	parser.add_argument('--k', type=int, dest='k', default=16, help='Number of eigenpairs to compute.');
	parser.add_argument('-v', action='store_true', dest='verbose', default=False, help='Runs program verbosely.')
	parser.add_argument('-d', '--debug', action='store_true', dest='debug', default=False, help='Prints debugging help.')
	
	values = parser.parse_args()
	global val
	val = values
	if val.debug: val.verbose = True
	
	assert( val.aff != 'Null' );
	
	A = np.load(val.aff);
	A = sp.csc_matrix((A['data'], A['indices'], A['indptr']), shape=A['shape']);

	timing.log('Matrix accepted, beginning NCuts...');
	
	ncuts(A, val, val.k);


