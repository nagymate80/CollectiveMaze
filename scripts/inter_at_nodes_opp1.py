#!/usr/bin/python

'''
Interaction finishes not definitely with out of new. Finishes when both leader and new have gone out.
'''
import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt
from math import cos, sin, pi, sqrt
from os.path import isfile
from igraph import *
from random import randint
from os import system
import fnmatch
import os
import copy

from time import time
time1=time()

run=sys.argv[1]
dnkrit = float(sys.argv[2])
mfix_notallowed = int(sys.argv[3])

def cosd(x):
    return cos(x*pi/180)
def sind(x):
    return sin(x*pi/180)

runstart_frame=-1

reach_target = {}
reach_target_prevedge = {}
with open('./runs/'+run+'/'+run+'_reach_target_manualstart.txt','r') as reach_target_file:
    for lindex, line in enumerate(reach_target_file):
        if lindex == 0:
            runstart_frame=int(line.split()[1])
            continue
        lin = line.split('\t')
        reach_target[ lin[0] ] = int(lin[2])
        reach_target_prevedge[ lin[0] ] = int(lin[1])
        
g=Graph()
#f=[ [ [796,312],[904,362] ],   [[904,362],[1014,304]], [[843,454],[976,460]]   ]
g.add_vertices(52)
with open('edges.txt', 'r') as efil:
    for esor in efil:
        node1=int(esor.split()[0])
        node2=int(esor.split()[1])
        g.add_edges([(node1,node2)])


with open('nodes.txt', 'r') as nfil:
    for nsor in nfil:        
        node=int(nsor.split()[0])
        
        x=int(nsor.split()[1])
        y=int(nsor.split()[2])                           
        g.vs[node]['x']=x
        g.vs[node]['y']=y
        
        

for index, edge in enumerate(g.get_edgelist()):
    x1=g.vs[edge[0]]['x']
    y1=g.vs[edge[0]]['y']
    x2=g.vs[edge[1]]['x']
    y2=g.vs[edge[1]]['y']
    
    g.es[index]['l']=sqrt((x2-x1)**2+(y2-y1)**2)
    


level=[
    [0],
    [1,2,49],
    [3,4,50,51],
    [5,6,7,8],
    [9,10,11,12,13,14,15,16],
    [17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
    [33,34,35,36,37,38,38,40,41,42,43,44,45,46,47,48]
    ]


for levelindex, levellist in enumerate(level):
    for node in levellist:
        g.vs[node]['level']=levelindex
        
#Help for calculating distances, x and y are vertex ids, d0 real dist in pixels, dn  normed (1 edge=1), dng graph representation
ee={x:{ y:{'d0':0, 'dn':0, 'dng':0, 'sign1':0, 'sign2':0} for y in range(0, len(g.es)) } for x in range(0, len(g.es))}

#halfs=[g.get_eid(0,1),g.get_eid(0,2),g.get_eid(1,3),g.get_eid(2,4)]

egeszek=[g.get_eid(3,5),g.get_eid(3,6),g.get_eid(4,7),g.get_eid(4,8),g.get_eid(5,9),g.get_eid(5,10),g.get_eid(6,11),g.get_eid(6,12),g.get_eid(7,13),g.get_eid(7,14),g.get_eid(8,15),g.get_eid(8,16)]


for x, edge1 in enumerate(g.get_edgelist()):
    for y, edge2 in enumerate(g.get_edgelist()):
        n11=min(edge1)
        n12=max(edge1)
        n21=min(edge2)
        n22=max(edge2)
        epath=g.get_shortest_paths(n11, to=n21, weights=None, mode=ALL, output="epath")[0]
        vpath=g.get_shortest_paths(n11, to=n21, weights=None, mode=ALL, output="vpath")[0]
        for lol in epath:
            ee[x][y]['d0']+=g.es[lol]['l']
            if lol in egeszek: #if lol not in halfs:
                ee[x][y]['dng']+=1
            else:
                ee[x][y]['dng']+=0.5
            ee[x][y]['dn']+=1
        if n12 in vpath:
            ee[x][y]['sign1']=-1
        else:
            ee[x][y]['sign1']=1
        if n22 in vpath:
            ee[x][y]['sign2']=-1
        else:
            ee[x][y]['sign2']=1
  
        
#print ee[g.get_eid(11,21)][g.get_eid(11,22)]['d0']
#print g.get_shortest_paths(11, to=11, weights=None, mode=ALL, output="epath")[0]

targets_nodes={ \
    1:[33,17,9],\
    2:[34,18,9],\
    3:[35,19,10],\
    4:[36,20,10],\
    5:[37,21,11],\
    6:[38,22,11],\
    7:[39,23,12],\
    8:[40,24,12],\
    9:[41,25,13],\
    10:[42,26,13],\
    11:[43,27,14],\
    12:[44,28,14],\
    13:[45,29,15],\
    14:[46,30,15],\
    15:[47,31,16],\
    16:[48,32,16]\
    }





def dist(e1,d1,e2,d2):
    if e1!=e2:
        return ee[e1][e2]['d0']+ee[e1][e2]['sign1']*d1+ee[e1][e2]['sign2']*d2
    else:
        return abs(d1-d2)

def distn(e1,d1n,e2,d2n):
    if e1!=e2:
        return ee[e1][e2]['dn']+ee[e1][e2]['sign1']*d1n+ee[e1][e2]['sign2']*d2n
    else:
        return abs(d1n-d2n)


def distng(e1,d1ng,e2,d2ng):
    if e1!=e2:
        return ee[e1][e2]['dng']+ee[e1][e2]['sign1']*d1ng+ee[e1][e2]['sign2']*d2ng
    else:
        return abs(d1ng-d2ng)

def lr_vect(px,py,x1,y1,x2,y2): #left or right side of the vector
    px,py,x1,y1,x2,y2=float(px),float(py),float(x1),float(y1),float(x2),float(y2)
    if (x2-x1)!=0:
        m=(y2-y1)/(x2-x1)
        ma=abs(m)
    else:
        ma=float('inf')
        m=float('inf')

    if 1<=ma: 
        if y2<y1:
            if px<=(py-y1)/m+x1:
                return 'left'
            else:
                return 'right'
        else:
            if px>=(py-y1)/m+x1:
                return 'left'
            else:
                return 'right'
 
    else: 
        if x2>x1:
            if py<=y1+m*(px-x1):
                return 'left'
            else:
                return 'right'
        else:
            if py>=y1+m*(px-x1):
                return 'left'
            else:
                return 'right'

def linelen(x1,y1,x2,y2):
    return sqrt((x2-x1)**2+(y2-y1)**2)

def linelensq(x1,y1,x2,y2):
    return (x2-x1)**2+(y2-y1)**2


for node in g.vs:
    node['numneigh']=len(g.neighbors(node.index))


starmids=[]

for s in g.vs:
    neighs=g.neighbors(s.index)
    s['alfa']=g.vs[neighs[0]]
    if len(neighs)>1:
        s['beta']=g.vs[neighs[1]]      
        if len(neighs)>2:
            s['gamma']=g.vs[neighs[2]]
            starmids.append(s.index)
            
starmids.remove(49)
starmids.remove(0)
starmids=sorted(starmids)



projfile='./runs/'+run+'/'+run+'_proj'



target = int(run.split('target')[1])
target_node = targets_nodes[target][0]
target_edge = g.get_eid( targets_nodes[target][0], targets_nodes[target][1] )
target_prev_edge = g.get_eid( targets_nodes[target][1], targets_nodes[target][2] )


for s in g.vs:
    if s.index == target_node:
        continue
    to_water_epath=g.get_shortest_paths(s.index, to=target_node, weights=None, mode=ALL, output="epath")[0]
    to_water_vpath=g.get_shortest_paths(s.index, to=target_node, weights=None, mode=ALL, output="vpath")[0]
    s['tw_e'] = to_water_epath[0]
    s['tw_n'] = to_water_vpath[1]
    if s['tw_n'] == s['alfa'].index:
        s['tw_abg'] = 'alfa'
    elif s['tw_n'] == s['beta'].index:
        s['tw_abg'] = 'beta'  
    elif s['tw_n'] == s['gamma'].index:
        s['tw_abg'] = 'gamma'


M12=['ORBM','OGBM','OBGM','GRBM','GRPM','GPBM','BRPM','BGPM']
M34=['ORPM','OBPM','OPGM','OPBM','GOBM','GOPM','GBPM','BOPM']
F12=['ROGF','ROBF','RGOF','RGBF','RGPF','ORBF','OGPF','GRBF']
F34=['ROPF','RBOF','RBGF','RBPF','RPOF','RPGF','RPBF','ORGF']

M12_lg=['ORBM','OGBM','OBGM','GRPM','BRPM','BGPM']
M34_lg=['ORPM','OPGM','GOBM','GOPM','GBPM','BOPM']
F12_lg=['ROGF','ROBF','RGOF','RGBF','RGPF','ORBF']
F34_lg=['RBOF','RBGF','RBPF','RPOF','RPGF','ORGF']

osszpatek=1

if run[0:5] == 'group':
    runtype = 'group'
    groupname = run.split('_target', 1)[0][-3:]
    gender=groupname[0]
    osszpatek=8

elif (run[0:5]=='learn') and (len(run.split('_')[2])==3):
    runtype = 'learng'
    groupname = run.split('_')[2]+'_lg'
    gender=groupname[0]
    osszpatek=6

elif (run[0:5]=='learn') and (len(run.split('_')[2])==4):
    runtype = 'learns'
    ratname = run.split('_')[2]
    gender=ratname[3]

elif run[0:6]=='single':
    runtype = 'single'
    ratname = run.split('_')[2]
    gender=ratname[3]


pateknames = []

if osszpatek == 1:
    pateknames = [ratname]
    
else:
    pateknames = eval(groupname)


ratpoz={}
ratpoz_prev={}

stars = { n:{} for n in starmids }
stars_prev = {}

for n in starmids:
    stars[n]['here'] = []
    stars[n]['num_here'] = 0
    stars[n]['interf'] = 0
    stars[n]['patek1'] = ''
    stars[n]['patek2'] = ''
    stars[n]['patek1_in'] = -1
    stars[n]['patek1_in_orient'] = 0
    stars[n]['patek1_isdrunk'] = 0
    stars[n]['patek1_out'] = -1
    stars[n]['patek1_out_orient'] = 0
    stars[n]['patek1_when_patek2_in'] = -1
    stars[n]['patek1_when_patek2_in_orient'] = 0
    stars[n]['patek2_in'] = -1
    stars[n]['patek2_in_orient'] = 0
    stars[n]['patek2_isdrunk'] = 0
    stars[n]['patek2_out'] = -1
    stars[n]['patek2_out_orient'] = 0

    stars[n]['vect_alfa'] = [(g.vs[n]['alfa']['x'] - g.vs[n]['x']) / g.es[g.get_eid(n,g.vs[n]['alfa'].index)]['l'] , \
                            (g.vs[n]['alfa']['y'] - g.vs[n]['y']) / g.es[g.get_eid(n,g.vs[n]['alfa'].index)]['l'] ]
    stars[n]['vect_beta'] = [(g.vs[n]['beta']['x'] - g.vs[n]['x']) / g.es[g.get_eid(n,g.vs[n]['beta'].index)]['l'] , \
                            (g.vs[n]['beta']['y'] - g.vs[n]['y']) / g.es[g.get_eid(n,g.vs[n]['beta'].index)]['l'] ]
    stars[n]['vect_gamma'] = [(g.vs[n]['gamma']['x'] - g.vs[n]['x']) / g.es[g.get_eid(n,g.vs[n]['gamma'].index)]['l'] , \
                            (g.vs[n]['gamma']['y'] - g.vs[n]['y']) / g.es[g.get_eid(n,g.vs[n]['gamma'].index)]['l'] ]


def is_abg(n1, n2):
    if g.vs[n1]['alfa'].index == n2:
        return 'alfa'
    elif g.vs[n1]['beta'].index == n2:
        return 'beta'
    elif g.vs[n1]['gamma'].index == n2:
        return 'gamma'
    else:
        return 'error'

def pairnode(node, color):
    n1n2 = [ ratpoz[color]['n1'], ratpoz[color]['n2'] ]
    if n1n2[0] != node:
        return n1n2[0]
    else:
        return n1n2[1]

def pairnode_prev(node, color):
    n1n2 = [ ratpoz_prev[color]['n1'], ratpoz_prev[color]['n2'] ]
    if n1n2[0] != node:
        return n1n2[0]
    else:
        return n1n2[1]

    
def check_in_star(node, color):
    if ratpoz[color]['n1'] == node:
        if ratpoz[color]['n1'] > ratpoz[color]['n2']:
            if ratpoz[color]['dnorm'] > (1-dnkrit):
                return True
            else:
                return False
        else: #n1<n2
            if ratpoz[color]['dnorm'] < dnkrit:
                return True
            else:
                return False

    elif ratpoz[color]['n2'] == node:
        if ratpoz[color]['n1'] > ratpoz[color]['n2']:
            if ratpoz[color]['dnorm'] < dnkrit:
                return True
            else:
                return False
        else: #n1<n2
            if ratpoz[color]['dnorm'] > (1-dnkrit):
                return True
            else:
                return False
        
    
    else:
        return False

def hither_away(node, color):
    vect_ori_x = cosd(ratpoz[color]['ori'])
    vect_ori_y = sind(ratpoz[color]['ori'])
    n1n2 = [ ratpoz[color]['n1'], ratpoz[color]['n2'] ]
    if n1n2[0] != node:
        pair_node = n1n2[0]
    else:
        pair_node = n1n2[1]
    #dotprod = vect_ori_x * (g.vs[pair_node]['x'] - g.vs[node]['x']) \
    #          + vect_ori_y * (g.vs[pair_node]['y'] - g.vs[node]['y'])
    #print node, pair_node
    abg=is_abg(node, pair_node)
    if abg=='error':
        return 0
    dotprod = vect_ori_x * stars[node]['vect_'+abg][0] \
              + vect_ori_y * stars[node]['vect_'+abg][1]
    if dotprod > 0:
        return 1 #away
    else:
        return -1 #hither

def hither_away_prev(node, color):
    vect_ori_x = cosd(ratpoz_prev[color]['ori'])
    vect_ori_y = sind(ratpoz_prev[color]['ori'])
    n1n2 = [ ratpoz_prev[color]['n1'], ratpoz_prev[color]['n2'] ]
    if n1n2[0] != node:
        pair_node = n1n2[0]
    else:
        pair_node = n1n2[1]
    #dotprod = vect_ori_x * (g.vs[pair_node]['x'] - g.vs[node]['x']) \
    #          + vect_ori_y * (g.vs[pair_node]['y'] - g.vs[node]['y'])
    #print node, pair_node
    abg=is_abg(node, pair_node)
    if abg=='error':
        return 0
    dotprod = vect_ori_x * stars[node]['vect_'+abg][0] \
              + vect_ori_y * stars[node]['vect_'+abg][1]
    if dotprod > 0:
        return 1 #away
    else:
        return -1 #hither


def one_patek_to_empty(n, color):
    stars[n]['interf'] = 1
    stars[n]['patek1'] = color
    stars[n]['patek1_in'] = is_abg(n, pairnode(n, color))
    stars[n]['patek1_in_orient'] = hither_away(n, color)
    if framenum > reach_target[color] + runstart_frame:
        stars[n]['patek1_isdrunk'] = True
    else:
        stars[n]['patek1_isdrunk'] = False

def second_patek_enters(n, color):
    stars[n]['interf'] = 2
    stars[n]['patek2'] = color
    stars[n]['patek2_in'] = is_abg(n, pairnode(n, color))
    stars[n]['patek2_in_orient'] = hither_away(n, color)
    if framenum > reach_target[color] + runstart_frame:
        stars[n]['patek2_isdrunk'] = True
    else:
        stars[n]['patek2_isdrunk'] = False
    stars[n]['patek1_when_patek2_in'] = is_abg(n, pairnode(n, stars[n]['patek1']))
    stars[n]['patek1_when_patek2_in_orient'] = hither_away(n, stars[n]['patek1'])
        
def one_patek_out(n, color):
    if stars[n]['interf'] in [2,3]:
        pid=1
        for pid_prob in [1,2]:
            if color == stars[n]['patek'+str(pid_prob)]:
                pid = pid_prob
                break
        stars[n]['patek'+str(pid)+'_out'] = is_abg(n, pairnode_prev(n, color))
        stars[n]['patek'+str(pid)+'_out_orient'] = hither_away_prev(n, color)


def del_patek12(n):
    stars[n]['interf'] = 0
    stars[n]['patek1'] = ''
    stars[n]['patek2'] = ''
    stars[n]['patek1_in'] = -1
    stars[n]['patek1_in_orient'] = 0
    stars[n]['patek1_isdrunk'] = 0
    stars[n]['patek1_out'] = -1
    stars[n]['patek1_out_orient'] = 0
    stars[n]['patek1_when_patek2_in'] = -1
    stars[n]['patek1_when_patek2_in_orient'] = 0
    stars[n]['patek2_in'] = -1
    stars[n]['patek2_in_orient'] = 0
    stars[n]['patek2_isdrunk'] = 0
    stars[n]['patek2_out'] = -1
    stars[n]['patek2_out_orient'] = 0

def write_event(n):
    interout.write('\t'.join([str(framenum),str(n),stars[n]['patek1'],stars[n]['patek2'],str(stars[n]['patek1_in']),\
                              str(stars[n]['patek1_in_orient']),str(stars[n]['patek1_out']),str(stars[n]['patek1_out_orient']),\
                              str(stars[n]['patek1_when_patek2_in']), str(stars[n]['patek1_when_patek2_in_orient'])   ]))
    interout.write('\t'+'\t'.join([str(stars[n]['patek2_in']),str(stars[n]['patek2_in_orient']),str(stars[n]['patek2_out']),\
                                   str(stars[n]['patek2_out_orient']),str(stars[n]['patek1_isdrunk']),str(stars[n]['patek2_isdrunk']), \
                                   str(g.vs[n]['tw_abg'])     ]))
    interout.write('\n')
    num_int[0] += 1

interout = open('./runs/'+run+'/'+run+'_inteactions_stars_opp_dnkrit_'+str(dnkrit)+'_mfix_'+str(mfix_notallowed)+'.txt','w')
interout.write('frame node patek1 patek2 patek1_in patek1_in_orient patek1_out patek1_out_orient patek1_when_patek2_in patek1_when_patek2_in_orient patek2_in patek2_in_orient patek2_out patek2_out_orient patek1_isdrunk patek2_isdrunk to_water\n')

num_int = [0]

with open(projfile, 'r') as data:
    
    osszframe=0

    for ind, line in enumerate(data):
        if ind==1:
            osszframe=int(line.split()[4])
        if ind < 15:
            continue
        
        sor=line.split()
        framenum=int(sor[0])
        pateknum=int(sor[1])

        ratpoz_prev=copy.deepcopy(ratpoz)
        stars_prev=copy.deepcopy(stars)

        ratpoz = {}

        for p in range(pateknum):
            color = sor[2+p*15]
            mfix = int(sor[16+p*15])
            if ((mfix & mfix_notallowed) == 0):
                if (color not in ratpoz.keys()):
                
                    ratpoz[color] = {}

                    ratpoz[color]['xp'] = float(sor[5+p*15])
                    ratpoz[color]['yp'] = float(sor[6+p*15])
                    ratpoz[color]['edge'] = int(sor[7+p*15])
                    ratpoz[color]['dnorm'] = float(sor[9+p*15])
                    ratpoz[color]['n1'] = int(sor[11+p*15])
                    ratpoz[color]['n2'] = int(sor[12+p*15])
                    ratpoz[color]['ori'] = float(sor[15+p*15])
                                                   
                else:
                    del ratpoz[color]
                    ratpoz[color] = {}

        for color in ratpoz.keys():
            if len(ratpoz[color].keys()) == 0:
                del ratpoz[color]
            

        for n in starmids:
            stars[n]['here'] = []
            for color in ratpoz.keys():
                if check_in_star(n, color):
                    stars[n]['here'].append(color)
            
            stars[n]['num_here'] = len(stars[n]['here'])
            if stars[n]['num_here'] > 2:
                if stars[n]['interf'] != 0:
                    del_patek12(n)

            elif stars[n]['num_here'] == 2:
                if stars_prev[n]['num_here'] == 0:
                    one_patek_to_empty(n, stars[n]['here'][0])
                    second_patek_enters(n, stars[n]['here'][1])
                elif stars[n]['interf'] == 1:
                    if stars[n]['patek1'] not in stars[n]['here']:
                        del_patek12(n)
                    else:
                        for color in stars[n]['here']:
                            if color not in stars_prev[n]['here']:
                                second_patek_enters(n, color)
                                break
                            
                elif stars[n]['interf'] == 2:
                    for color in stars[n]['here']:
                        if color not in stars_prev[n]['here']:
                            del_patek12(n)
                            break

                elif stars[n]['interf'] == 3:
                    del_patek12(n)
                    
            elif stars[n]['num_here'] == 1:
                if stars_prev[n]['num_here'] == 0:
                    one_patek_to_empty(n, stars[n]['here'][0])
                elif stars_prev[n]['num_here'] == 1:
                    if stars[n]['here'][0] != stars_prev[n]['here'][0]:
                        del_patek12(n)
                elif stars[n]['interf'] == 2:
                    for color in stars_prev[n]['here']:
                        if color not in stars[n]['here']:
                            one_patek_out(n, color)
                            stars[n]['interf'] = 3

                    
            elif stars[n]['num_here'] == 0:
                if stars[n]['interf'] == 3:
                    one_patek_out(n, stars_prev[n]['here'][0])
                    write_event(n)

                del_patek12(n)


                   
                        
    


interout.close()

time2=time()
runtime=time2-time1
print "Runtime: ", runtime, run
#print dnkrit, num_int











    
    
