
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
import sys

#==============================================================================
# This is a python translation of 'Downsampled Normalized Cuts', an algorithm
# created by researchers at UC Berkely for image segmentation.
#==============================================================================
def save_sparse_csc(filename,array):
	np.savez(filename, data = array.data, indices=array.indices, indptr=array.indptr, shape=array.shape )

def to_array(a):
	shape = 0;
	for i in a: shape += len(i);
	array = np.zeros(shape, dtype='int64');
	
	c = 0;
	for i in range(len(a)):
		temp = np.array(a[i]);
		array[c:c+temp.shape[0]] = temp;
		c += temp.shape[0];
	
	return array;

def sparse_truncate(a, thresh):
	a.data = np.multiply((a.data > thresh),a.data);
	a.eliminate_zeros();

def update_progress(progress):
    barLength = 40 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

def sparse_multiply(a,b,n,val):
	assert( a.getformat() == 'csr' );
	assert( b.getformat() == 'csc' );
	#	Function to multiply large sparse matrices w/o memory errors and print density

	c = [];
	#	a = [a_1 ... a_n]^T	
	data = [];
	indices = [];
	indptr = [];
	submat_shape = np.array([a.shape[0]//n,b.shape[1]//n]);
	metarow = [];
	if val.verbose: timing.log('Beginning Multiplication');
	for i in range(n):
		metarowlen = 0;
		for j in range(n):
			a_lower = (a.shape[0]//n)*i;
			a_upper = (a.shape[0]//n)*(i+1);
			if i == n-1: a_upper = a.shape[0];

			b_lower = (b.shape[1]//n)*j;
			b_upper = (b.shape[1]//n)*(j+1);
			if j == n-1: b_upper = b.shape[1];

			a_i = a[a_lower:a_upper];
			b_i = b[:, b_lower:b_upper];

			c = sp.csr_matrix.dot(a_i,b_i);
			shp = len(c.data);
			metarowlen += shp;

			sparse_truncate(c[-1], 0.00001);	#	In-method truncation to limit dense construction

			sub_ind = np.memmap('.memmapped/spmultiply/%i_%iindices.array' %(i,j), dtype='int64', mode='w+', shape=shp);
			sub_data = np.memmap('.memmapped/spmultiply/%i_%idata.array' %(i,j), dtype='float64', mode='w+', shape=shp);
			sub_indptr = np.memmap('.memmapped/spmultiply/%i_%iindptr.array' %(i,j), dtype='int64', mode='w+', shape=(submat_shape[0]+1));
			sub_ind[:] = c.indices.astype('int64');
			sub_data[:] = c.data.astype('float64');
			sub_indptr[:] = c.indptr.astype('int64');
			sub_ind.flush(); sub_data.flush(); sub_indptr.flush();
			del c, sub_ind, sub_data, sub_indptr;
			update_progress( (j+i*n+1) / float(n**2) );
		metarow.append(metarowlen);
	if val.verbose: timing.log('Multiplication complete, beginning reassembly.');
	#	Reassemble
	metaindptr = [];
	start = 0;
	end = 0;
	ind = np.memmap('.memmapped/spmultiply/metarowindices.array', dtype='int64', mode='w+', shape=np.sum(np.array(metarow)));
	data = np.memmap('.memmapped/spmultiply/metarowdata.array', dtype='float64', mode='w+', shape=np.sum(np.array(metarow)));
	for i in range(n):		
		indptr = [];
		indptr.append(0);
		for k in range(submat_shape[0]):
			rowlen = 0;
			for j in range(n):
				start = end;
				sub_ind = np.memmap('.memmapped/spmultiply/%i_%iindices.array' %(i,j), dtype='int64', mode='r+');
				sub_data = np.memmap('.memmapped/spmultiply/%i_%idata.array' %(i,j), dtype='float64', mode='r+');
				sub_indptr = np.memmap('.memmapped/spmultiply/%i_%iindptr.array' %(i,j), dtype='int64', mode='r+');
				
				rowdata = sub_data[sub_indptr[k]:sub_indptr[k+1]];
				rowind = sub_ind[sub_indptr[k]:sub_indptr[k+1]];
				
				datalen = rowdata.shape[0];
				end = start + datalen;
				rowlen += datalen;

				ind[start:end] = (rowind+submat_shape[1]*j).astype('int64');
				
				data[start:end] = rowdata.astype('float64');
				update_progress( (j+k*n+n*i*(submat_shape[0])+1) / float((n**2)*submat_shape[0]) );
			indptr.append(indptr[-1]+rowlen);
		if i == 0: metaindptr.append(indptr);
		else: metaindptr.append(metaindptr[-1][-1]+np.array(indptr[1:]));
		ind.flush(); data.flush();
	
	indptr = to_array(metaindptr);	#supplementary function
	aff = sp.csc_matrix((data[:], ind[:], indptr.astype('int64')), shape=(a.shape[0],a.shape[0]));
	del sub_ind, sub_data, sub_indptr, rowdata, rowind, indptr, metaindptr, b_i, a_i, a,b;
	return aff;

def half_sparse(a):
	ind = np.memmap('tempindices.array', dtype='int64', mode ='w+', shape=a.indices.shape);
	indptr = np.memmap('tempindptr.array', dtype='int64', mode ='w+', shape=np.ceil(a.shape[0]/2)+1);
	data = np.memmap('tempdata.array', dtype='float64', mode ='w+', shape=a.data.shape);
	bottom = 0;
	indptr[0] = np.float64(0);
	for i in range(0,a.indptr.shape[0]-a.indptr.shape[0]%2,2):
		di = a.indptr[i+1] - a.indptr[i];
		#print di;
		top = bottom + di;
		ind[bottom:top] = a.indices[a.indptr[i]:a.indptr[i+1]];
		data[bottom:top] = a.data[a.indptr[i]:a.indptr[i+1]];
		indptr[i//2+1] = np.float64(top);
		bottom = top;
	#return sp.csc_matrix( (data[:top], ind[:top], indptr[:]), shape=(a.shape[1],np.ceil(a.shape[0]/2)) )
	return sp.csr_matrix( (data[:top], ind[:top], indptr[:]), shape=(np.ceil(a.shape[0]/2),a.shape[1]) )
		

def dncuts(a, opts, config, nvec=16, n_downsample=2, decimate=2):

# a = affinity matrix
# nevc = number of eigenvectors (set to 16?)
# n_downsample = number of downsampling operations (2 seems okay)
# decimate = amount of decimation for each downsampling operation (set to 2)
	
	
	a_down=sp.csc_matrix(a);
	n = config['spmult_blocksize'];		#	Blocks in sparse multiplication
	list1 = [None] * n_downsample
	
	#Loop Start =========================================
	for di in range(n_downsample):
		
#Decimate sparse matrix
		a_sub = half_sparse(a_down);
		if opts.debug: print a_sub.shape
		if opts.verbose: timing.log('Downsampled sparse matrix #%s' % str(di+1))
		#a_sub = a_sub.transpose();

# Normalize the downsampled affinity matrix using C-code parallelized with OpenMP
		if opts.verbose: timing.log('Normalizing matrix...')
		a_tmp = a_sub.tocsc();
		d = array(a_tmp.sum(0)).reshape(a_tmp.shape[1])
		if opts.debug: np.save('savefiles/tmp_indptr.npy', a_tmp.indptr); np.save('savefiles/tmp_data.npy', a_tmp.data); 
		norm(d.size, d, a_tmp.indptr, a_tmp.data)
		if opts.verbose: timing.log('Matrix Normalized')
		b = a_tmp.transpose().tocsc();

# "Square" the affinity matrix, while downsampling
		if opts.debug: print a_sub.shape, b.shape
		
		if opts.verbose: timing.log('Before dot-product')
		if opts.debug: print a_sub.nnz, b.nnz;
		if opts.debug: print np.sum(a_sub.data > 0.0001);
		del a_tmp, d; 
		print a_sub.getformat(), b.getformat();
		a_down = sparse_multiply(a_sub, b, n, opts)
		#sparse_truncate(a_down, 0.000001)
		if opts.verbose: timing.log('After dot-product')
		
		
		if opts.debug: print 'adown', a_down.shape
		if opts.debug and opts.save: save_sparse_csc('adown%i.npz'%(di), a_down);
  
# Hold onto the normalized affinity matrix for upsampling later
		list1[di]=b
		if opts.verbose: timing.log('End of loop #%s' % str(di+1))
	#Loop end ============================================  
	
# Get the eigenvectors
	if opts.verbose: timing.log('Starting NCuts...')
	del a_sub, b;
	if opts.debug: save_sparse_csc('savefiles/a_down.npz',a_down);
	cuteigs = ncuts(a_down, opts, k=nvec)
	if opts.verbose: timing.log('Finished NCuts')
	Ev = cuteigs.ev
	Eval = cuteigs.evl

	if opts.debug and opts.save: np.save('eigenvectors.npy', Ev);
	if opts.debug and opts.save: np.save('eigenvalues.npy', Eval);	

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
	
