#!/usr/bin/python
import sys
from igraph import *
from math import cos, sin, pi, sqrt
from copy import deepcopy
from os import system

dnkrit = 0.3
mfix_notallowed = 0



g=Graph()
g.add_vertices(52)
with open('/home/attila/PATEKL/video/edgecount/edges_2.csv', 'r') as efil:
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


egeszek=[g.get_eid(3,5),g.get_eid(3,6),g.get_eid(4,7),g.get_eid(4,8),g.get_eid(5,9),g.get_eid(5,10),g.get_eid(6,11),g.get_eid(6,12),g.get_eid(7,13),g.get_eid(7,14),g.get_eid(8,15),g.get_eid(8,16)]

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

res = {}
for node in starmids:
    res[node] = {'beta_num':0,'gamma_num':0,'beta_p':0,'gamma_p':0}

def is_abg(n1, n2):
    if g.vs[n1]['alfa'].index == n2:
        return 'alfa'
    elif g.vs[n1]['beta'].index == n2:
        return 'beta'
    elif g.vs[n1]['gamma'].index == n2:
        return 'gamma'
    else:
        return 'error'

    
with open('/home/attila/PATEKL/video/edgecount/dirnames/dirnames_group','r') as runs:
    for run in runs:
        run = run[:-1]
        
        with open('/home/attila/PATEKL/video/edgecount/runs/'+run+'/'+run+'_individuals_stars_v6_dnkrit_'+str(dnkrit)+'_mfix_'+str(mfix_notallowed)+'.txt','r') as inter:
            for l in inter:
                if l[0] == 'f':
                    continue
                s = l.split('\t')

                fin = s[3]
                if fin != 'alfa':
                    continue

                fout = s[5]

                if fout == 'alfa':
                    continue

                node = int(s[1])

                res[node][fout+'_num'] += 1


for node in res:
    for f in ['beta','gamma']:
        res[node][f+'_p'] = float(res[node][f+'_num']) / float(res[node]['beta_num'] + res[node]['gamma_num'])


with open('/home/attila/PATEKL/video/right_left/beta_gamma.dat','w') as dat:
    for index, edge in enumerate(g.get_edgelist()):
        gx1=g.vs[edge[0]]['x']
        gy1=g.vs[edge[0]]['y']
        gx2=g.vs[edge[1]]['x']
        gy2=g.vs[edge[1]]['y']
        dgx=gx2-gx1
        dgy=gy2-gy1
        if edge[0] in starmids:
            dat.write(str(gx1)+'\t'+str(gy1)+'\t'+str(dgx)+'\t'+str(dgy)+'\t'+str(res[edge[0]][is_abg(edge[0], edge[1])+'_p'])+'\n')
        elif edge[1] not in [49, 50, 51]:
            if edge[0] == 0:
                dat.write(str(gx1)+'\t'+str(gy1)+'\t'+str(dgx)+'\t'+str(dgy)+'\t'+'0.5'+'\n')
            else:
                dat.write(str(gx1)+'\t'+str(gy1)+'\t'+str(dgx)+'\t'+str(dgy)+'\t'+'1'+'\n')


system("gnuplot -e \"infile='/home/attila/PATEKL/video/right_left/beta_gamma.dat'; outfile='/home/attila/PATEKL/video/right_left/beta_gamma.jpg'\" /home/attila/PATEKL/video/scripts_plt/faplot_color.plt")                
                


