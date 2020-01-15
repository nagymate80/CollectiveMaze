import numpy
import math
from random import random

dnkrit = 0.3
mfix_notallowed = 0

#to_partner_ratio = []
#event_nums = []
to_partner = 0
to_other = 0

abg_id = {'alfa':0, 'beta':1, 'gamma':2}

def rlb(back, dir2): #right-left-back
    if back == 'alfa':
        if dir2 == 'alfa':
            return 'back'
        elif dir2 == 'beta':
            return 'right'
        elif dir2 == 'gamma':
            return 'left'

    elif back == 'beta':
        if dir2 == 'alfa':
            return 'left'
        elif dir2 == 'beta':
            return 'back'
        elif dir2 == 'gamma':
            return 'right'

    elif back == 'gamma':
        if dir2 == 'alfa':
            return 'right'
        elif dir2 == 'beta':
            return 'left'
        elif dir2 == 'gamma':
            return 'back'

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = numpy.average(values, weights=weights)
    variance = numpy.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return (average, math.sqrt(variance))

def bootstrap(values, weights, k, num_choice):
    sumw = sum(weights)
    cum = []
    for index, i in enumerate(weights):
        cum.append(float(i)/float(sumw))
        if index > 0:
            cum[index] += cum[index-1]

    avs = []
    #print v
    #print w
    #print cum
   

    for nn in xrange(num_choice):
        avs.append(0.0)
        for kk in xrange(k):
            r = random()
            #print 'r', r
            for index, c in enumerate(cum):
                 if c > r:
                     #print 'chosen', v[index]
                     avs[nn] += float(values[index]) / float(k)
                     break
                 
    return (numpy.average(avs), numpy.std(avs), avs)





with open('/home/hattila/Patek/atl_ratmaze/video/edgecount/dirnames/dirnames_group','r') as runs:
    for run in runs:
        run = run[:-1]

        #to_partner = 0
        #to_other = 0
        
        with open('/home/hattila/Patek/atl_ratmaze/video/edgecount/runs/'+run+'/'+run+'_inteactions_stars_v2_0.3.txt','r') as inter:        
            for l in inter:
                if (l[0] == 'f') or (l[0] == '0'):
                    continue
                s = l[:-1].split('\t')

                focal_in_edge = s[12]
                partner_edge = s[8]
                partner_orient = s[9]
                focal_out_edge = s[14]
                focal_isdrunk = s[18]
                partner_isdrunk = s[17]
                to_water = s[19]
                focal_beta_ago = s[21]
                focal_gamma_ago = s[22]
                node = s[1]

                if focal_in_edge != 'alfa':
                    continue
                if partner_edge != 'beta':
                    continue
                if partner_orient != '-1':
                    continue
                if focal_out_edge == 'alfa':
                    continue
                if focal_isdrunk != 'False':
                    continue
                if partner_isdrunk != 'False': 
                    continue
                if to_water != 'alfa':
                    continue
                #if focal_beta_ago != '-1':
                #    continue
                #if focal_gamma_ago != '-1':
                #    continue
                #if node != '4':
                #    continue

                
                if focal_out_edge == partner_edge:
                    to_partner += 1
                else:
                    to_other += 1
        '''
        events = to_partner + to_other
        if events > 0:
            to_partner_ratio.append(float(to_partner)/float(events))
            event_nums.append(events)
        '''



#for index, i in enumerate(to_partner_ratio):
#    print i, event_nums[index]
#print weighted_avg_and_std(to_partner_ratio, event_nums)

print to_partner, to_other

print bootstrap([1, 0], [to_partner, to_other], (to_partner+to_other), 1000)[0:2]








        
