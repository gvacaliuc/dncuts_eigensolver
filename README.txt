golena.py is a code that will find the approximate eigenvectors and eigenvalues of an image or more directly an affinity matrix.  It is set up to find the eigens of 'lena.bmp', which is found in the source directory. If you edit golena.py, you can have it calculate/visualize the eigens of any image you pass it. visual.py is a standalone code that visualizes saved eigenvector (.npy) files.

NOTE:  Dncuts can be run on it's own for images, but using this repo with protein conformational clustering requires the infrastructure built in 'dQCuts-pipeline'.

	Run: '$ python golena.py -h/--help' to get a feel for the flags of the code.  

DEPENDENCIES:
	I have included any 'non-normal' dependencies or modules that I wrote in the module packages. However this is a list of the dependencies python must have installed for this code to run:
	Numpy
	Matplotlib
	Scipy

C-Extension: In /packages you will find a sub-directory labeled 'c-extensions'. This is the source for _norm_c.so, the f2py wrapped parallel c code used in dncuts.py. There is no need to run, _norm_c.so is independent, I have just included it for your convenience.

Everything else should be installed with python stock. Email me @
gabe.vacaliuc@gmail.com if you have any questions, comments, or bug fixes.
Have fun!

NOTE: Argparse was introduced new in Python 2.7, so be sure that your python is compatible.
NOTE: The timing module uses a python module that is deprecated in Python 3, so be sure that your python is compatible.

This program is a python implementation of the 'dncuts' method/algorithm highlighted in the first paper:
http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/mcg/
