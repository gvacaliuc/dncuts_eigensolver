import numpy as np
import matplotlib.pyplot as plt
def visualize(evf, evt, shape, nvec):
	print 'Visualizing...'
	vistrue = evt.reshape(shape[0], shape[1], -1)
	visfast = evf.reshape(shape[0], shape[1], -1)
	vistrue = vistrue[:,:,:nvec]
	visfast = visfast[:,:,:nvec]
	
	vistrue = 4 * np.sign(vistrue) * np.abs(vistrue)**(1/2)
	visfast = 4 * np.sign(visfast) * np.abs(visfast)**(1/2)

	vistrue = np.maximum(0, np.minimum(1, vistrue))
	visfast = np.maximum(0, np.minimum(1, visfast))
	#if val.debug: print 'vistrue: ', vistrue.shape, 'visfast: ', visfast.shape
	g,h,l = vistrue.shape
	m = np.floor(np.sqrt(l))
	n = np.ceil(l/m)
	mont_true = np.zeros((g*m, h*n))
	mont_fast = np.zeros((g*m, h*n))
	#if val.debug: print mont_true.shape

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
			#if val.debug: print count
	plt.figure()
	plt.subplot(2,1,2)
	plt.subplots_adjust(bottom=.1)
	plt.title('True eigenvectors: 1 - %s' % nvec)
	plt.imshow(mont_true)
	plt.subplot(2,1,1)
	plt.title('Fast eigenvectors: 1 - %s' % nvec)
	plt.imshow(mont_fast)
	plt.show()

import scipy.misc as misc
def compare(evf, img):
	print 'Visualizing...'
	shape = [256,256]
	plt.figure()
	
	im = np.zeros(shape)
	ev = evf
	ev = ev.reshape(shape[0], shape[1], -1)
	ev = ev.transpose(1,0,2)
	for i in range(16):
		im = im + ev[:,:,i]
	plt.subplot(2,1,1)
	plt.title('True Eigenvectors vs Image \n')
	plt.imshow(im)
	plt.subplot(2,1,2)
	plt.imshow(img)
	plt.show()


#                     slot 1                    slot 2
visualize(np.load('rough_eigv.npy'), np.load('rough_eigv.npy'), [320,405], 16)
compare(np.load('rough_eigv.npy'), misc.imread('kensenter.jpg'))

