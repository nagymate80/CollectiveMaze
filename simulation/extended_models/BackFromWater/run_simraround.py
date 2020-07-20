#!/usr/bin/python
# -*- coding: utf-8 -*-

# to run: python simraround.py 8 4 0.0 0.0 1.0 0.0

#import sys
#from os import system

import subprocess

#import simraroundNEW

# print 'Give n, level, h, m, s, vis'
# level=int(sys.argv[2])	# number of hierarchical levels of the maze. possible endpoints will be 2^level
# n=int(sys.argv[1])	# number of individauls searching
# h=float(sys.argv[3])	# parameter for the probability of turning back. (by default should be 0)
# m=float(sys.argv[4])	# parameter for the weighting based on the number of visited endpoints
# s=float(sys.argv[5])	# social attraction towards edges where individuals are
# vis=float(sys.argv[6])	# parameter for the weighting of the last visit

#number of steps an individual need to make to get to the next juntion
instepsMin=2
instepsMax=3

write_traj = 0


#for vis_i in range(-4,8,1):
#    if vis_i == -4:
#	vis = 0
#    else:
#	vis = 2**(vis_i)
#    
#    for s_i in range(-4,8,1):
#        if s_i == -4:
#		s = 0
#	else:
#	    s = 2**(s_i)
#
#	subprocess.call('python simraroundNEW.py 8 4 0 0.0 %g %g >>test_out' %(s,vis), shell=True)

for m_i in range(-4,8,1):
    if m_i == -4:
	m = 0
    else:
	m = 2**(m_i)
    
    for s_i in range(-4,8,1):
        if s_i == -4:
		s = 0
	else:
	    s = 2**(s_i)

	subprocess.call('python simraroundNEW.py 8 4 0.0 %g %g 0.0 >>test_out' %(m,s), shell=True)

