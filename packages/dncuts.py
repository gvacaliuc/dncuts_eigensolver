from __future__ import division
import numpy as np
from numpy import *
from scipy.sparse.linalg import *
import collections
from whiten import *
from ncuts import *
import timing
import scipy.sparse as sp
import matplotlib.pyplot as plt
from _norm_c import *

#==============================================================================
# This is a python translation of 'Downsampled Normalized Cuts', an algorithm
# created by researchers at UC Berkely for image segmentation.
#==============================================================================

def dncuts(a, opts, nvec=16, n_downsample=2, decimate=2):

# a = affinity matrix
# nevc = number of eigenvectors (set to 16?)
# n_downsample = number of downsampling operations (2 seems okay)
# decimate = amount of decimation for each downsampling operation (set to 2)
	
	
	a_down=sp.csc_matrix(a);
	list1 = [None] * n_downsample
	
	#Loop Start =========================================
	for di in range(n_downsample):
		
#Decimate sparse matrix
		g,h = a_down.shape;
		idx = (np.arange(h) % 2).astype('bool');
		a_sub = a_down[:,idx];
		if opts.debug: print a_sub.shape
		if opts.verbose: timing.log('Downsampled sparse matrix #%s' % str(di+1))
		a_sub = a_sub.transpose()

# Normalize the downsampled affinity matrix using C-code parallelized with OpenMP
		if opts.verbose: timing.log('Normalizing matrix...')
		a_tmp = a_sub.tocsc()
		d = array(a_tmp.sum(0)).reshape(a_tmp.shape[1])
		norm(d.size, d, a_tmp.indptr, a_tmp.data)
		if opts.verbose: timing.log('Matrix Normalized')
		b = a_tmp.transpose()

# "Square" the affinity matrix, while downsampling
		if opts.debug: print a_sub.shape, b.shape
		
		if opts.verbose: timing.log('Before dot-product')
		a_down = sp.csc_matrix(a_sub.dot(b))
		if opts.verbose: timing.log('After dot-product')
		
		if opts.debug: print 'adown', a_down.shape
  
# Hold onto the normalized affinity matrix for upsampling later
		list1[di]=b
		if opts.verbose: timing.log('End of loop #%s' % str(di+1))
	#Loop end ============================================  
	
# Get the eigenvectors
	if opts.verbose: timing.log('Starting NCuts...')
	cuteigs = ncuts(a_down, opts, k=nvec)
	if opts.verbose: timing.log('Finished NCuts')
	Ev = cuteigs.ev
	Eval = cuteigs.evl

#Some debugging help	
	if opts.debug: print 'list', list1[0].shape
	if opts.debug: print 'list', list1[1].shape
	if opts.debug: print 'ev', Ev.shape
	if opts.debug: print 'eval', Eval.shape, Eval

# Upsample the eigenvectors
	if opts.verbose: timing.log('Upsampling eigenvectors...')
	for di in range(n_downsample-1,-1,-1):
		Ev = list1[di].dot(Ev)
		if opts.debug: print 'ev', Ev.shape
	if opts.verbose: timing.log('Eigenvectors upsampled')

# "Upsample" the eigenvalues
	if opts.debug: print Eval
	Eval = 1./(2.**(n_downsample))*Eval
	if opts.debug: print Eval
	if opts.verbose: timing.log('\'Upsampled\' eigenvalues')

# whiten the eigenvectors
	if opts.verbose: timing.log('Whitening eigenvectors...')
	Ev = whiten(Ev, opts)
	if opts.verbose: timing.log('Eigenvectors whitened')
	
	eigens = collections.namedtuple('eigens', ['ev','evl'])
	eig = eigens(ev=Ev.real, evl=Eval.real)
	if opts.debug: print 'ev', Ev.shape
	return eig
	
