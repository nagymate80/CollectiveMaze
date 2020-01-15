#In this script I use the reciprocal of the reach frame
from os import system
import numpy

system('mkdir targets_real')

real_telj = {}
single_telj = {}
gr_telj = {}
sr_telj = {}

for t in range(1,17):
    real_telj[t] = []
    single_telj[t] = []
    gr_telj[t] = []
    sr_telj[t] = []

dirnames = ['dirnames_group_h1', 'dirnames_single80', 'randomruns_group_sametarget', 'randomruns_single_sametarget']

for did, dirname in enumerate(dirnames):
    dfile = open('/home/attila/PATEKL/video/edgecount/dirnames/'+dirname,'r')
    if did in [0,2,3]:
        osszpatek=8
    else:
        osszpatek=1
    
    for line1 in dfile:
        run=line1[:-1]
        runsplit=run.split('_')
        target = int(runsplit[3][6:])
        #roundd = runsplit[0][5:]
        if did in [0,1]:
            reachfile = open('/home/attila/PATEKL/video/edgecount/runs/'+run+'/'+run+'_reach_target_manualstart.txt','r')
        else:
            reachfile = open('/media/EC5616105615DBE0/Users/Attila/Documents/random/runs/'+run+'/'+run+'_reach_target_manualstart.txt','r')
        

        reach_frames = []

        
        for line2 in reachfile:
            if line2[0] == '#':
                continue
            if line2[-1] == '\n':
                line2 = line2[:-1]
                
            if line2.split('\t')[2] == '0':
                print run
                break
            if line2.split('\t')[2] != '-1':
                reach_frames.append(float(line2.split('\t')[2]))
            else:
                reach_frames.append(90000)
                
        else:
            if did == 0:
                real_telj[target].append(numpy.average(reach_frames))
            elif did == 1:
                single_telj[target].append(numpy.average(reach_frames))
            elif did == 2:
                gr_telj[target].append(numpy.average(reach_frames))
            elif did == 3:
                sr_telj[target].append(numpy.average(reach_frames))
        reachfile.close()



    dfile.close()

for t in range(1,17):
    print t, numpy.average(real_telj[t]), numpy.std(real_telj[t]), numpy.average(single_telj[t]), numpy.std(single_telj[t]), numpy.average(gr_telj[t]), numpy.std(gr_telj[t]), numpy.average(sr_telj[t]), numpy.std(sr_telj[t])
