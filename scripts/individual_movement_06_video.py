#!/usr/bin/python
import sys
from igraph import *
from math import cos, sin, pi, sqrt
from copy import deepcopy
from os import system
from time import time
time1=time()

'''
examine later:
-right/left/back
-to_water
-least_remember
'''

run=sys.argv[1]
dnkrit = 0.3 #float(sys.argv[2])
mfix_notallowed = 784 #int(sys.argv[3])

albega = ['alfa','beta','gamma']

def cosd(x):
    return cos(x*pi/180)
def sind(x):
    return sin(x*pi/180)

g=Graph()
g.add_vertices(52)
with open('/home/attila/PATEKL/video/edgecount/edges.txt', 'r') as efil:
    for esor in efil:
        node1=int(esor.split()[0])
        node2=int(esor.split()[1])
        g.add_edges([(node1,node2)])


with open('/home/attila/PATEKL/video/edgecount/nodes_2.txt', 'r') as nfil:
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

ee={x:{ y:{'d0':0, 'dn':0, 'dng':0, 'sign1':0, 'sign2':0} for y in range(0, len(g.es)) } for x in range(0, len(g.es))}
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

ends_edges = {}
for end in range(1,17):
    ends_edges[end] = g.get_eid(targets_nodes[end][0], targets_nodes[end][1])


for node in g.vs:
    node['numneigh']=len(g.neighbors(node.index))

starmids=[]

for s in g.vs:
    neighs=g.neighbors(s.index)
    s['alfa']=g.vs[neighs[0]]
    s['alfa_edge'] = g.get_eid(s.index, s['alfa'].index)
    if len(neighs)>1:
        s['beta']=g.vs[neighs[1]]
        s['beta_edge'] = g.get_eid(s.index, s['beta'].index)
        if len(neighs)>2:
            s['gamma']=g.vs[neighs[2]]
            s['gamma_edge'] = g.get_eid(s.index, s['gamma'].index)
            starmids.append(s.index)
            
starmids.remove(49)
starmids.remove(0)
starmids=sorted(starmids)



projfile='/home/attila/PATEKL/video/edgecount/runs/'+run+'/'+run+'_proj'

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

pateks = {}
pateknames = []

if osszpatek == 1:
    pateknames = [ratname]
    
else:
    pateknames = eval(groupname)

for color in pateknames:
    pateks[color] = {}
    pateks[color]['interf'] = 0
    pateks[color]['n'] = -1
    pateks[color]['in'] = -1
    pateks[color]['in_or'] = 0
    pateks[color]['out'] = -1
    pateks[color]['out_or'] = 0    
    pateks[color]['alfa.focal_ago_n'] = -1
    pateks[color]['beta.focal_ago_n'] = -1
    pateks[color]['gamma.focal_ago_n'] = -1
    pateks[color]['alfa.partner_ago_n'] = -1
    pateks[color]['beta.partner_ago_n'] = -1
    pateks[color]['gamma.partner_ago_n'] = -1
    pateks[color]['alfa.anybody_ago_n'] = -1
    pateks[color]['beta.anybody_ago_n'] = -1
    pateks[color]['gamma.anybody_ago_n'] = -1
    pateks[color]['isdrunk'] = False
    pateks[color]['is_sy_drinking_when_new_in'] = 0
    pateks[color]['tw'] = 0
    pateks[color]['alfa_num']=0
    pateks[color]['beta_num']=0
    pateks[color]['gamma_num']=0
    pateks[color]['closest_partner_norm']=-1
    

ratpoz={}
ratpoz_prev={}

rat_epop={} #number of frames when rat was on the edge
rat_epop2={} #number of times when rat was on the edge
rat_epop2_from_alfa={}

rat_edge_last = {color:[ -1 for edge in g.es()] for color in pateks.keys()}
        
rat_id = {}

for p, ratname in enumerate(pateknames):
    rat_id[ratname] = p
    rat_epop[ratname] =  [0 for edge in g.es()]
    rat_epop2[ratname] =  [0 for edge in g.es()]

def how_many_rats_visited_edge(edge):
    res = 0
    for color in pateknames:
        if rat_epop[color][edge] > 0:
            res += 1
    return res

def how_many_times_rats_visited_edge(edge):
    res = 0
    for color in pateknames:
        if rat_epop2[color][edge] > 0:
            res += rat_epop2[color][edge]
    return res

def how_many_other_rats_visited_edge(color, edge):
    res = 0
    for partner in pateknames:
        if (partner != color) and (rat_epop[partner][edge] > 0):
            res += 1
    return res

def how_many_times_other_rats_visited_edge(color, edge):
    res = 0
    for partner in pateknames:
        if (partner != color) and (rat_epop2[partner][edge] > 0):
            res += rat_epop2[partner][edge]
    return res

stars = { n:{} for n in starmids }
for n in starmids:
    stars[n]['vect_alfa'] = [(g.vs[n]['alfa']['x'] - g.vs[n]['x']) / g.es[g.get_eid(n,g.vs[n]['alfa'].index)]['l'] , \
                            (g.vs[n]['alfa']['y'] - g.vs[n]['y']) / g.es[g.get_eid(n,g.vs[n]['alfa'].index)]['l'] ]
    stars[n]['vect_beta'] = [(g.vs[n]['beta']['x'] - g.vs[n]['x']) / g.es[g.get_eid(n,g.vs[n]['beta'].index)]['l'] , \
                            (g.vs[n]['beta']['y'] - g.vs[n]['y']) / g.es[g.get_eid(n,g.vs[n]['beta'].index)]['l'] ]
    stars[n]['vect_gamma'] = [(g.vs[n]['gamma']['x'] - g.vs[n]['x']) / g.es[g.get_eid(n,g.vs[n]['gamma'].index)]['l'] , \
                            (g.vs[n]['gamma']['y'] - g.vs[n]['y']) / g.es[g.get_eid(n,g.vs[n]['gamma'].index)]['l'] ]
    stars[n]['end_edges_alfa'] = []
    stars[n]['end_edges_beta'] = []
    stars[n]['end_edges_gamma'] = []
    for end in targets_nodes:
        way_to_end = g.get_shortest_paths(n, to=targets_nodes[end][0], weights=None, mode=ALL, output="vpath")[0]
        if g.vs[n]['alfa'].index in way_to_end:
            stars[n]['end_edges_alfa'].append(ends_edges[end])
        elif g.vs[n]['beta'].index in way_to_end:
            stars[n]['end_edges_beta'].append(ends_edges[end])        
        elif g.vs[n]['gamma'].index in way_to_end:
            stars[n]['end_edges_gamma'].append(ends_edges[end])    

def check_in_big_star(node, color):
    if node in g.get_edgelist()[ratpoz[color]['edge']]:
        return True
    else:
        return False


def check_in_small_star(node, color):
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

def is_abg(n1, n2):
    if g.vs[n1]['alfa'].index == n2:
        return 'alfa'
    elif g.vs[n1]['beta'].index == n2:
        return 'beta'
    elif g.vs[n1]['gamma'].index == n2:
        return 'gamma'
    else:
        return 'error'

def pairnode(color, node):
    n1n2 = [ ratpoz[color]['n1'], ratpoz[color]['n2'] ]
    if n1n2[0] != node:
        return n1n2[0]
    else:
        return n1n2[1]

def pairnode_prev(color, node):
    n1n2 = [ ratpoz_prev[color]['n1'], ratpoz_prev[color]['n2'] ]
    if n1n2[0] != node:
        return n1n2[0]
    else:
        return n1n2[1] 

def hither_away(color, node):
    vect_ori_x = cosd(ratpoz[color]['ori'])
    vect_ori_y = sind(ratpoz[color]['ori'])
    n1n2 = [ ratpoz[color]['n1'], ratpoz[color]['n2'] ]
    if n1n2[0] != node:
        pair_node = n1n2[0]
    else:
        pair_node = n1n2[1]
    abg=is_abg(node, pair_node)
    if abg=='error':
        return 0
    dotprod = vect_ori_x * stars[node]['vect_'+abg][0] \
              + vect_ori_y * stars[node]['vect_'+abg][1]
    if dotprod > 0:
        return 1 #away
    else:
        return -1 #hither



def hither_away_prev(color, node):
    vect_ori_x = cosd(ratpoz_prev[color]['ori'])
    vect_ori_y = sind(ratpoz_prev[color]['ori'])
    n1n2 = [ ratpoz_prev[color]['n1'], ratpoz_prev[color]['n2'] ]
    if n1n2[0] != node:
        pair_node = n1n2[0]
    else:
        pair_node = n1n2[1]

    abg=is_abg(node, pair_node)
    if abg=='error':
        return 0
    dotprod = vect_ori_x * stars[node]['vect_'+abg][0] \
              + vect_ori_y * stars[node]['vect_'+abg][1]
    if dotprod > 0:
        return 1 #away
    else:
        return -1 #hither

'''
def w_entry_params(n, color):
    pateks[color]['in'] = is_abg(n, pairnode(color,n))
    pateks[color]['in_orient'] = hither_away(color, n)
'''

def is_sy_drinking():
    for color in ratpoz.keys():
        if ratpoz[color]['edge'] == target_edge:
            return True
            break
    else:
        return False

def edge_focalago(color, n1, n2):
    if rat_edge_last[color][g.get_eid(n1,n2)] != -1:
        return framenum - rat_edge_last[color][g.get_eid(n1,n2)]
    else:
        return -1

def edge_anybodyago(n1,n2):
    res = -1
    for color in pateknames:
        if edge_focalago(color, n1, n2) > -1:
            if (res == -1) or (edge_focalago(color, n1, n2) < res):
                res = edge_focalago(color, n1, n2)
    return res

def edge_partnerago(color, n1,n2):
    res = -1
    for partner in pateknames:
        if (partner != color) and (edge_focalago(partner, n1, n2) > -1):
            if (res == -1) or (edge_focalago(partner, n1, n2) < res):
                res = edge_focalago(partner, n1, n2)
    return res

def write_event(n, color):
    if (color not in ratpoz) or (color not in ratpoz_prev):
        return -1
    pateks[color]['out'] = is_abg(n, pairnode_prev(color, n))
    pateks[color]['out_or'] = hither_away_prev(color, n)
    interout.write('\n')
    interout.write('\t'.join([str(framenum), str(pateks[color]['n']), \
        color, pateks[color]['in'], str(pateks[color]['in_or']),pateks[color]['out'], str(pateks[color]['out_or']), \
                              str(pateks[color]['is_sy_drinking_when_new_in']), str(pateks[color]['isdrunk']), \
                              g.vs[pateks[color]['n']]['tw_abg'], str(pateks[color]['alfa.anybody_ago']),str(pateks[color]['beta.anybody_ago']),str(pateks[color]['gamma.anybody_ago']), str(pateks[color]['alfa_num']),str(pateks[color]['beta_num']),str(pateks[color]['gamma_num']),str(pateks[color]['alfa_vis']),str(pateks[color]['alfa_notvis']),str(pateks[color]['beta_vis']),str(pateks[color]['beta_notvis']),str(pateks[color]['gamma_vis']),str(pateks[color]['gamma_notvis']),\
                              str(pateks[color]['alfa.focal_ago']),str(pateks[color]['beta.focal_ago']),str(pateks[color]['gamma.focal_ago']),str(pateks[color]['alfa.partner_ago']),str(pateks[color]['beta.partner_ago']),str(pateks[color]['gamma.partner_ago']),str(pateks[color]['closest_partner_norm']),\
                              str(how_many_other_rats_visited_edge(color, g.vs[n]['alfa_edge'])), str(how_many_other_rats_visited_edge(color, g.vs[n]['beta_edge'])),str(how_many_other_rats_visited_edge(color, g.vs[n]['gamma_edge'])), \
                              str(how_many_times_other_rats_visited_edge(color, g.vs[n]['alfa_edge'])), str(how_many_times_other_rats_visited_edge(color, g.vs[n]['beta_edge'])), str(how_many_times_other_rats_visited_edge(color, g.vs[n]['gamma_edge']))           ]))
    

def entry(n, color):
    pateks[color]['interf'] = 1
    pateks[color]['n'] = n
    pateks[color]['in'] = is_abg(n, pairnode(color,n))
    pateks[color]['in_or'] = hither_away(color, n)
    pateks[color]['alfa.focal_ago'] = edge_focalago(color, n, g.vs[n]['alfa'].index)
    pateks[color]['beta.focal_ago'] = edge_focalago(color, n, g.vs[n]['beta'].index)
    pateks[color]['gamma.focal_ago'] = edge_focalago(color, n, g.vs[n]['gamma'].index)
    pateks[color]['alfa.partner_ago'] = edge_partnerago(color, n, g.vs[n]['alfa'].index)
    pateks[color]['beta.partner_ago'] = edge_partnerago(color, n, g.vs[n]['beta'].index)
    pateks[color]['gamma.partner_ago'] = edge_partnerago(color, n, g.vs[n]['gamma'].index)
    pateks[color]['alfa.anybody_ago'] = edge_anybodyago(n, g.vs[n]['alfa'].index)
    pateks[color]['beta.anybody_ago'] = edge_anybodyago(n, g.vs[n]['beta'].index)
    pateks[color]['gamma.anybody_ago'] = edge_anybodyago(n, g.vs[n]['gamma'].index)
    pateks[color]['isdrunk'] = ratpoz[color]['isdrunk']
    pateks[color]['is_sy_drinking_when_new_in'] = is_sy_drinking()
    for partner in ratpoz.keys():
        if partner == color:
            continue
        if (ratpoz[partner]['n1'] in [n, g.vs[n]['alfa'].index]) and (ratpoz[partner]['n2'] in [n, g.vs[n]['alfa'].index]):
            pateks[color]['alfa_num']+=1
        elif (ratpoz[partner]['n1'] in [n, g.vs[n]['beta'].index]) and (ratpoz[partner]['n2'] in [n, g.vs[n]['beta'].index]):
            pateks[color]['beta_num']+=1
        elif (ratpoz[partner]['n1'] in [n, g.vs[n]['gamma'].index]) and (ratpoz[partner]['n2'] in [n, g.vs[n]['gamma'].index]):
            pateks[color]['gamma_num']+=1
    for albi in albega:
        pateks[color][albi+'_vis'] = 0
        pateks[color][albi+'_notvis'] = 0
        for endedge in stars[n]['end_edges_'+albi]:
            if rat_epop[color][endedge] > 0:
                pateks[color][albi+'_vis'] += 1
            else:
                pateks[color][albi+'_notvis'] += 1
    pateks[color]['closest_partner_norm']= closest_partner_norm(color)
    #pateks[color]['tw_abg'] see: g.vs[n]['tw_abg']    


def del_entry_params(n, color):
    pateks[color]['interf'] = 0
    pateks[color]['n'] = -1
    pateks[color]['in'] = -1
    pateks[color]['in_or'] = 0
    pateks[color]['out'] = -1
    pateks[color]['out_or'] = 0    
    pateks[color]['alfa.focal_ago_n'] = -1
    pateks[color]['beta.focal_ago_n'] = -1
    pateks[color]['gamma.focal_ago_n'] = -1
    pateks[color]['alfa.partner_ago_n'] = -1
    pateks[color]['beta.partner_ago_n'] = -1
    pateks[color]['gamma.partner_ago_n'] = -1
    pateks[color]['alfa.anybody_ago_n'] = -1
    pateks[color]['beta.anybody_ago_n'] = -1
    pateks[color]['gamma.anybody_ago_n'] = -1
    pateks[color]['is_sy_drinking_when_new_in'] = 0
    pateks[color]['alfa_num']=0
    pateks[color]['beta_num']=0
    pateks[color]['gamma_num']=0     
    pateks[color]['closest_partner_norm']=-1

def closest_partner_norm(color):
    closest = -1
    for partner in ratpoz.keys():
        if (partner != color):
            dist_color_partner = distn(ratpoz[color]['edge'], ratpoz[color]['dnorm'], ratpoz[partner]['edge'], ratpoz[partner]['dnorm'])
            if ( (closest==-1) or (dist_color_partner < closest) ):
                closest = dist_color_partner
    return closest
            


reach_target = {}
reach_target_prevedge = {}
with open('/home/attila/PATEKL/video/edgecount/runs/'+run+'/'+run+'_reach_target_manualstart.txt','r') as reach_target_file:
    for lindex, line in enumerate(reach_target_file):
        if lindex == 0:
            continue
        lin = line.split('\t')
        reach_target[ lin[0] ] = int(lin[2])
        reach_target_prevedge[ lin[0] ] = int(lin[1])


interout = open('/home/attila/PATEKL/video/edgecount/runs/'+run+'/'+run+'_individuals_stars_v6_dnkrit_'+str(dnkrit)+'_mfix_'+str(mfix_notallowed)+'.txt','w')
interout.write('\t'.join(['frame','node','color','in','in_orient','out','out_orient','sy_drinking','isdrunk',\
                          'to_water','alfa.anybodyago','beta.anybodyago','gamma.anybodyago','alfa_num','beta_num','gamma_num',\
                          'alfa_vis','alfa_notvis','beta_vis','beta_notvis','gamma_vis','gamma_notvis',\
                          'alfa.focalago','beta.focalago','gamma.focalago','alfa.partnerago','beta.partnerago','gamma.partnerago',\
                          'closest_partner_distn', 'alfa_how_many_other_rats_vis', 'beta_how_many_other_rats_vis', 'gamma_how_many_other_rats_vis',\
                          'alfa_how_many_times_others_vis','beta_how_many_times_others_vis','gamma_how_many_times_others_vis']))
    
with open(projfile, 'r') as data:
    
    osszframe=0
    runstart_frame=-1
    with open('/home/attila/PATEKL/video/edgecount/runstart/summarycoding14list.csv', 'r') as runstartfile:
        for line in runstartfile:
            lin = line.split('\t')
            if (lin[0] == run) and (lin[1] == "start"):
                runstart_frame = 1500*int(lin[2]) + 25*int(lin[3])

    
    for ind, line in enumerate(data):
        if ind==1:
            osszframe=int(line.split()[4])
        if ind < 15:
            continue
        if ind == 1500:
            break
        sor=line.split()
        framenum=int(sor[0])
        pateknum=int(sor[1])

        ratpoz_prev=deepcopy(ratpoz)
        ratpoz = {}

        rats = open('/home/attila/PATEKL/video/edgecount/rats.txt', 'w')
        circles = open('/home/attila/PATEKL/video/edgecount/circles.txt', 'w')


        for p in range(pateknum):
            color = sor[2+p*15]
            mfix = int(sor[16+p*15])
            if ((mfix & mfix_notallowed) == 0):
                if (color not in ratpoz.keys()):
                
                    ratpoz[color] = {}

                    ratpoz[color]['xp'] = float(sor[5+p*15])
                    ratpoz[color]['yp'] = float(sor[6+p*15])
                    ratpoz[color]['edge'] = int(sor[7+p*15])
                    rat_epop[color][ ratpoz[color]['edge'] ] += 1
                    if (color not in ratpoz_prev.keys()) or (ratpoz[color]['edge'] != ratpoz_prev[color]['edge']):
                        rat_epop2[color][ ratpoz[color]['edge'] ] += 1
                    ratpoz[color]['dnorm'] = float(sor[9+p*15])
                    ratpoz[color]['n1'] = int(sor[11+p*15])
                    ratpoz[color]['n2'] = int(sor[12+p*15])
                    ratpoz[color]['ori'] = float(sor[15+p*15])
                    ratpoz[color]['mfix'] = mfix
                   
                    if framenum > reach_target[color] + runstart_frame:
                        ratpoz[color]['isdrunk'] = True
                    else:
                        ratpoz[color]['isdrunk'] = False

                else:
                    del ratpoz[color]
                    ratpoz[color] = {}

        for color in ratpoz.keys():
            if len(ratpoz[color].keys()) == 0:
                del ratpoz[color]
            else:
                rat_edge_last[color][ ratpoz[color]['edge'] ] = framenum

        for color in pateknames:
            if color not in ratpoz.keys():
                if pateks[color]['interf'] == 1:
                    del_entry_params(pateks[color]['n'], color)
            else:
                if pateks[color]['interf'] == 0:
                    if (ratpoz[color]['n1'] in starmids) and check_in_small_star(ratpoz[color]['n1'], color):
                        entry(ratpoz[color]['n1'], color)


                else: #pateks[color]['interf'] == 1
                    if not check_in_small_star(pateks[color]['n'], color):
                        write_event(pateks[color]['n'], color)
                        del_entry_params(pateks[color]['n'], color)
                        
                        if (ratpoz[color]['n1'] in starmids) and check_in_small_star(ratpoz[color]['n1'], color):
                            entry(ratpoz[color]['n1'], color)

                    else: #still here
                        closest_act = closest_partner_norm(color)
                        if (closest_act < pateks[color]['closest_partner_norm']) or (pateks[color]['closest_partner_norm'] == -1):
                            pateks[color]['closest_partner_norm'] = closest_act
                        

        
        for p, color in enumerate(ratpoz.keys()):
            rats.write(str(ratpoz[color]['xp'])+'\t'+str(ratpoz[color]['yp'])+'\t'+str(rat_id[color]))
            if p < len(ratpoz.keys()) -1:
                rats.write('\n')

        numhere = {n:0 for n in starmids}
        for color in pateknames:
            if pateks[color]['interf'] == 1:
                numhere[ pateks[color]['n'] ] += 1
        for n in starmids:
            if numhere[n] > 0:
                circles.write(str(g.vs[n]['x'])+'\t'+str(g.vs[n]['y'])+'\t'+str(numhere[n])+'\n')

        rats.close()
        circles.close()
        system('gnuplot -e "outfile=\'/home/attila/PATEKL/video/edgecount/runs/'+run+'/'+str(10000000+framenum)+'\'; pateks='+str(osszpatek)+'" /home/attila/PATEKL/video/scripts_plt/faplotg4.4.plt')
        


        #print framenum, '/', osszframe

interout.close()

system('mencoder "mf:///home/attila/PATEKL/video/edgecount/runs/'+run+'/10*.jpg" -mf fps=25 -o /home/attila/PATEKL/video/anim_videos/'+run+'_ind6.mpg -ovc lavc -lavcopts vcodec=mpeg4')
#system('rm /home/attila/PATEKL/video/edgecount/runs/'+run+'/10*.jpg')        

time2=time()
runtime=time2-time1
print "Runtime: ", runtime, run
                
