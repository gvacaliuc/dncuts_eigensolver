from __future__ import division
import numpy as np
from numpy import *
from scipy import misc, io
from packages.gaussaffinity import *
from packages.ncuts import *
from packages.dncuts import *
import matplotlib.pyplot as plt
import collections
import argparse
import packages.timing
import scipy.sparse as sp

#================================================================================================================================
# This is the driver code behind dncuts. In main() you'll find the image collection, where you can tailor the code to your image.
#================================================================================================================================
 
parser = argparse.ArgumentParser()
parser.add_argument('--im', type=str, default='lena.bmp', help='Image to perform cut on.')
parser.add_argument('--sz1', type=int, default=256, help='Length of image')
parser.add_argument('--sz2', type=int, default=256, help='Width of image')
parser.add_argument('-g', action='store_true', dest='graph', default=False, help='Shows graph of Affinity Matrix and eigenv\'s, depending on flags.')
parser.add_argument('-n', '--ncut', action='store_true', dest='ncut', default=False, help='Performs NCuts instead of DNCuts. Cannot be used with -c.')
parser.add_argument('-v', action='store_true', dest='verbose', default=False, help='Runs program verbosely.')
parser.add_argument('-s', '--save', action='store_true', dest='save', default=False, help='Saves affinity matrix and eigenv\'s.')
parser.add_argument('-c', '--comp', '--compare', action='store_true', dest='compare', default=False, help='Compares fast and true eigenv\'s.')
parser.add_argument('-d', '--debug', action='store_true', dest='debug', default=False, help='Prints debugging help.')
parser.add_argument('--vis', action='store_true', dest='visual', default=False, help='Visualizes both true and fast eigenvectors. Dependendent on -c.i')
parser.add_argument('im', type=str, help='The Image')
parser.add_argument('sz1', type=int, help='Image size 1')
parser.add_argument('sz2', type=int, help='Image size 2')
parser.add_argument('-p', action='store_true', dest='pic', default=False, help='Show picture of fast eigenvectors.')

values = parser.parse_args()
global val
val = values
if val.debug: val.verbose = True
cut = True
if val.compare: cut = False

#Little code to save sparse csc matrices
def save_sparse_csc(filename,array):
	np.savez(filename, data = array.data, indices=array.indices, indptr=array.indptr, shape=array.shape )
#End of save_sparse_csc

#Equivalent to Matlabs 'ismember' but only computes indices
def ismember(a,b):
	c = -np.ones(len(a))
	for i in range(len(a)):
		for j in range(len(b)):
			if c[i] != -1:
				continue
			if a[i] == b[j]:
				c[i] = j
	c[np.where(c == -1)] = 0
	return c
#End of ismember

#Compares the true eigens with fast eigens=======================================================================
def grapheigens(evfast, evalfast, evreal, evalreal, nvec=10):
	fig = plt.figure()
	eigenvalues = fig.add_subplot(111)
	eigenvalues.plot(np.arange(nvec), evalfast[:10], 'b-', np.arange(nvec), evalreal[:10], 'r-')
	eigenvalues.set_title('Eigenvalue comparison')
	eigenvalues.legend(('Blue = DNCuts', 'Red = NCuts'), loc='upper left')
	plt.show()
	fig1 = plt.figure()
	fig1.add_subplot(nvec,1,1)
		
	for i in range(nvec):
		plt.subplot(nvec,1,i+1)
		plt.yticks([])
		plt.xticks([])
		plt.plot(np.arange(len(evfast[:,i])), evfast[:,i], 'b-', np.arange(len(evreal[:,i])), evreal[:,i], 'r-')
		if i == nvec-1: plt.xticks(np.linspace(0,len(evfast[:,i]),1000))
	plt.suptitle('Eigenvector comparison')
	plt.legend(('Blue = DNCuts', 'Red = NCuts'), loc='lower right')
	plt.show()
#End of grapheigens =============================================================================================

#Code to visualize the eigenvectors, true and fast ==============================================================
def visualize(evf, evt, evl, im, nvec):
	if val.verbose: print 'Visualizing...'
	vistrue = evt.reshape(len(im[:,0,0]), len(im[0,:,0]), -1)
	visfast = evf.reshape(len(im[:,0,0]), len(im[0,:,0]), -1)
	vist = vistrue[:,:,:nvec]
	visf = visfast[:,:,:nvec]
	
	vistrue = 4 * np.sign(vist) * np.abs(vist)**(1/2)
	visfast = 4 * np.sign(visf) * np.abs(visf)**(1/2)

	vistrue = np.maximum(0, np.minimum(1, vistrue))
	visfast = np.maximum(0, np.minimum(1, visfast))
	if val.debug: print 'vistrue: ', vistrue.shape, 'visfast: ', visfast.shape
	g,h,l = vistrue.shape
	m = np.floor(np.sqrt(l))
	n = np.ceil(l/m)
	mont_true = np.zeros((g*m, h*n))
	mont_fast = np.zeros((g*m, h*n))
	if val.debug: print mont_true.shape

	#Construct montage
	count = 0
	for i in range(m.astype(int)):
		for j in range(n.astype(int)):
			try:
				mont_true[i*g:g+i*g,j*h:h+j*h] = vistrue[:,:,count] 
				mont_fast[i*g:g+i*g,j*h:h+j*h] = visfast[:,:,count]
			except:
				mont_true[i*g:g+i*g,j*h:h+j*h] = 0 
				mont_fast[i*g:g+i*g,j*h:h+j*h] = 0
			count = count + 1
			if val.debug: print count
	plt.figure()
	plt.subplot(2,1,1)
	plt.title('True eigenvectors: 1 - %s' % str(nvec))
	plt.imshow(mont_true)
	plt.subplot(2,1,2)
	plt.title('Fast eigenvectors: 1 - %s' % str(nvec))
	plt.imshow(mont_fast)
	
	plt.show() #shows the montage

	plt.figure()
	im = np.zeros((im.shape[0], im.shape[1]))
	ev = visf
	ev = ev.transpose(1,0,2)
	for i in range(nvec):
		im = im + ev[:,:,i]
	plt.imshow(im)

	plt.show() #shows the fast eigenvectors

#End of visualize ===============================================================================================

#Main============================================================================================================
def main(im='lena.bmp', sz1=256, sz2=256): 
	#Code to construct a gaussian affinity matrix of 'lena.bmp', perform DNCuts, and print eigen-v's
	xy = 7 #radius of search
	rgb_sigma = 30 #divide rgb differences by this
	nvec = 16
	ndown = 2
	
	#Import image and resize
	if val.verbose: print 'Importing and sizing image...'

	img = misc.imread(im)
	img = misc.imresize(img, (sz1,sz2))

	img = misc.imread(val.im)
	img = misc.imresize(img, (val.sz1, val.sz2))

	if val.verbose: print 'Image acquired'
	
	#Get the pixel affinity matrix and save it
	if val.verbose: timing.log('Constructing gaussian affinity matrix...')
	A = getgauss(img, val)
	if val.verbose: timing.log('Gaussian affinity matrix acquired')
	if val.save: save_sparse_csc('affinity_lena256.npy', A)
	
	#Loads matlab affinity matrix
	#A = np.load('A_affinitymat.npy.npz')
	#A = sp.csc_matrix((A['data'], (A['row'], A['col'])), shape=A['shape'])
	if val.debug: print A.shape, A
	
	#Graph the affinity matrix -- gets annoying when graphing real stuff...
	#if val.graph:
	#	if val.verbose: print 'Plotting affinity matrix...'
	#	fig = plt.figure()
	#	ax1 = fig.add_subplot(111)
	#	ax1.spy(A.todense()
	#	plt.show()
	
#Get the eigens the user wants
	if (val.ncut == True) and (val.compare==False):
		if val.verbose: timing.log('Starting straight NCuts...')
		true_eig = ncuts(A, val)
		if val.verbose: timing.log('Finished straight NCuts')
		if val.debug: print 'Eigenvectors:', true_eig.ev
		if val.debug: print 'Eigenvalues:', true_eig.evl
		if val.save: np.save('true_eigv.npy', true_eig.ev)
		if val.save: np.save('true_eigval.npy', true_eig.evl)
	
	elif (cut):
		if val.verbose: timing.log('Starting DNCuts...')
		fast_eig = dncuts(A, val)
		if val.verbose: timing.log('Finished DNCuts')
		if val.debug: print 'Eigenvectors:', fast_eig.ev
		if val.debug: print 'Eigenvalues:', fast_eig.evl
		if val.save: np.save('rough_eigv.npy', fast_eig.ev)
		if val.save: np.save('true_eigval.npy', fast_eig.evl)
	
	plt.figure()
	i = np.zeros((sz1, sz2))
	ev = fast_eig.ev
	ev = ev.transpose(1,0,2)
	for j in range(nvec):
		i = i + ev[:,:,i]
	plt.imshow(i)

	plt.show()
#Potential to call grapheigens
	if val.compare:
		if val.verbose: timing.log('Starting DNCuts...')
		fast_eig = dncuts(A, val, n_downsample = ndown)
		if val.verbose: timing.log('Finished DNCuts')
		
		if val.verbose: timing.log('Starting straight NCuts...')
		true_eig = ncuts(A, val)
		if val.verbose: timing.log('Finished straight NCuts')
		
		if val.graph: 
			print 'Comparison before eigenvector processing...'
			grapheigens(fast_eig.ev, fast_eig.evl, true_eig.ev, true_eig.evl, nvec=10)
	#Eigenvector clean up (reordering, resigning...)
		if val.verbose: print 'Cleaning up Eigenvectors...'
		EV_fast = fast_eig.ev
		EV_true = true_eig.ev
		C = np.abs(np.dot(EV_fast.T, EV_true))
		accuracy = np.trace(C)/nvec
		if val.verbose: print 'Accuracy of Eigenvector intially ', accuracy*100., '%'
		M = np.arange(len(C[:,0]))
		for p in range(10):
			M_last = M
			for i in range(len(C[:,0])):
				for j in range(i+1,len(C[:,0])):
					if (C[i,M[j]] + C[j,M[i]]) >  (C[i,M[i]] + C[j,M[j]]):
						m = M[j]
						M[j] = M[i]
						M[i] = m
			if np.all(M == M_last):
				break
		if val.debug: print 'M = ', M.shape, M
		M = ismember(np.arange(nvec, dtype=int), M.astype(int))
		
		EV_fast = EV_fast[:,M.astype(int)]
	
		sig = np.sign(np.sum(EV_fast*EV_true, 0))
		EV_fast = EV_fast*sig
		C = np.dot(EV_fast.transpose(), EV_true)
		accuracy = np.trace(C)/nvec
		if val.verbose: print 'Accuracy of Eigenvector after processing ', accuracy*100., '%'
		if val.save: np.save('fast_eigv.npy', EV_fast)
		if val.save: np.save('fast_eigval.npy', fast_eig.evl)
		if val.save: np.save('true_eigv.npy', true_eig.ev)
		if val.save: np.save('true_eigval.npy', true_eig.evl)
		if val.graph:
			print 'Comparison after eigenvector processing...'
			grapheigens(EV_fast, fast_eig.evl, true_eig.ev, true_eig.evl, nvec=10)
		
		if val.visual:
			visualize(EV_fast, EV_true, fast_eig.evl, img, nvec)
	if val.pic:
		e = np.dot(fast_eig.ev,fast_eig.evl)
		print e.shape
		plt.figure()
		plt.imshow(e.reshape(val.sz2,val.sz1).T)
		plt.show()
#End of Main ====================================================================================================

if __name__ == '__main__':
	main(val.im, val.sz1, val.sz2)
