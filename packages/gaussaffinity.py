from __future__ import division
import numpy as np
from numpy import *
from scipy import sparse as sp
import timing

#========================================================
# Finds the gaussian affinity between pixels in an image.
#========================================================

def getgauss(im, opts, xyrad=7, RGB_SIGMA=30):
	
	g, h, z = im.shape

	# Find all pairs of pixels within a distance of XY_RADIUS
	dj, di = np.meshgrid(np.arange(-xyrad, xyrad + 1), np.arange(-xyrad, xyrad + 1))
	dv = (dj**2 + di**2) <= xyrad**2
	
	di = di[dv]
	dj = dj[dv]
	
	i,j = np.meshgrid(np.arange(1,g+1), np.arange(1,h+1))

	m, n = i.shape
	i = i.reshape(m*n, 1)	
	i = np.tile(i, (1, len(di)))
	
	j = j.reshape(m*n, 1)	
	j = np.tile(j, (1, len(di)))
	
	itemp = i + di.transpose()
	jtemp = j + dj.transpose()
	vtmp = (itemp >= 1) & (itemp <= g) & (jtemp >= 1) & (jtemp <= h)
	
	helper = np.arange(g*h).reshape(g,h)

	pair_i = helper[i[vtmp]-1, j[vtmp]-1]
	pair_j = helper[itemp[vtmp]-1, jtemp[vtmp]-1]
	
	# Weight each pair by the difference in RGB values, divided by RGB_SIGMA
	if opts.debug: timing.log('check1')
	im = im.transpose(1,0,2)	
	RGB = im.reshape(-1,im.shape[2])/RGB_SIGMA
	if opts.debug: print 'RGB', RGB.shape, RGB
	w = np.exp(-np.sum((RGB[pair_i,:] - RGB[pair_j,:])**2.0,1))
	if opts.debug: print w.shape
	#Construct an affinity matrix
	if opts.debug: timing.log('check2')
	A = sp.csc_matrix((w, (pair_i, pair_j)), shape=(g*h, g*h))
	if opts.debug: print 'A', A.shape
	return A
