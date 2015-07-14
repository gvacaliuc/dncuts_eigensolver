import atexit
from time import clock

#==============
# Timing Module
#==============

global start

def begin():
	log("Start Program")
	global start
	start = clock()

def secondsToStr(t):
    return "%d:%02d:%02d.%03d" % \
        reduce(lambda ll,b : divmod(ll[0],b) + ll[1:],
            [(t*1000,),1000,60,60])

line = "="*40

def log(s, elapsed=None):
	print line
	print secondsToStr(clock()), '-', s
	if elapsed:
		print "Elapsed time:", elapsed
	print line
	print

def endlog():
	end = clock()
	global start
	try: 
		elapsed = end-start
		log("End Program", secondsToStr(elapsed))
	except:
		junk = 5	

def now():
	return secondsToStr(clock())


atexit.register(endlog)
