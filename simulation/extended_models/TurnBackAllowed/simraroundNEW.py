#!/usr/bin/python
# -*- coding: utf-8 -*-

# to run: python simraround.py 8 4 0.0 0.0 1.0 0.0

from igraph import *
from random import *
import sys
from os import system
from numpy import sign
import numpy as np


#### Parameters to set
write_traj = 0
iter_num = 1000


if len(sys.argv)!=7:
    print 'Give n, level, h, m, s, vis'
    sys.exit()
level=int(sys.argv[2])	# number of hierarchical levels of the maze. possible endpoints will be 2^level
n=int(sys.argv[1])	# number of individauls searching
h=float(sys.argv[3])	# parameter for the probability of turning back. (by default should be 0)
m=float(sys.argv[4])	# parameter for the weighting based on the number of visited endpoints
s=float(sys.argv[5])	# social attraction towards edges where individuals are
vis=float(sys.argv[6])	# parameter for the weighting of the last visit

#number of steps an individual need to make to get to the next juntion
instepsMin=2
instepsMax=3


'''
#Number of rats
n=8
#Constants
#forward preference
e=1 #ez fix
#back
h=0.0
# this is the weighting of the visited endpoints	old version: stay
m=0.0
#attraction
s=0.0

level=4
'''
e=1

def num_ver(level):
    ver=0
    for i in range(level+1):
        ver+=2**i
    return ver

def get_eid1(g, v1, v2):
    # get edge index based on v1<v2
    eid = g.get_eid( min(v1,v2), max(v1, v2) )
    return eid


def step_one(r, g, rats, goal, n, h, m, s, vis, step):
    neig={v:1 for v in g.neighbors(rats[r]['poz'])}
    ratio_unvis={}
    reltime_lastvis={}
    partners={}
    downwards_choices={}
    
    #neig[rats[r]['poz']]=1
    #for v in neig.keys():
    #	print "rat: "+ str(r)+ "v: "+ str(v) + "neig[v]: " + str(neig[v])
    osszcucc=0
    go_to=-1
    if rats[r]['steps_remaining']!=0:
	rats[r]['steps_remaining'] -= 1
    else:
        if rats[r]['poz']==goal:
	    go_to=rats[r]['poz']
	    rats[r]['steps_remaining']=0
	else:
	    rnd = random()
	    # decide on turning back:
	    if (rnd <= h) and (rats[r]['poz']<num_ver(level-1)) and (rats[r]['poz']>num_ver(1)):
		go_to = rats[r]['prev_poz']
	    else:
	    
    		for v in neig.keys():
        	    if v==rats[r]['prev_poz']: 
        		if rats[r]['poz']<num_ver(level-1):	# if at the deadend turning back count different way
            		    neig[v]=0	# turning back	#*s**len(g.vs[v]['here_are'])
        		else:
            		    neig[v]=neig[v]
            		    #print "deadend"

        	    else:
        		downwards_choices[v]=1
        		# if coming from the direction of the route, check number of unvisited endpoints
        		eid = get_eid1(g, rats[r]['poz'], v)
        		if rats[r]['prev_poz'] < rats[r]['poz']:
        		    visited_endpoints = 0
        		    unvisited_endpoints = 0
        		    for ep in g.vs[v]['reachable_endpoints']:
        			if r in g.vs[ep]['visited_by']:
        			    visited_endpoints += 1
        			else:
        			    unvisited_endpoints += 1
        		
        		    #print "r: " +str(r) + "\tvisited_endpoints: " + str(visited_endpoints) + "\tunvisited_endpoints: " + str(unvisited_endpoints)
        		    #print "on this edge: " + str(g.es[eid]['here_are'])
        		
        		    ratio_unvis[v]= 1.0*unvisited_endpoints/(visited_endpoints+unvisited_endpoints)
        		
        		    ###		    # weighting based on visited endpoints
        		    neig[v] = neig[v] * (1+m*ratio_unvis[v]) 
        		else:
            		    neig[v] = neig[v] 
            		
            		# weighting based on other rats on the edge	
            		partners[v] = sign(len(g.es[eid]['here_are']))
            		# weighting based on last visit to that node by the rat
    			reltime_lastvis[v] = (step-g.vs[v]['last_visit'][r])/(len(g.es)*0.5*(instepsMax+instepsMin))
    			neig[v] = neig[v] * (1+s*partners[v]) * (1+vis*reltime_lastvis[v])
    			#print "r: " + str(r) + "\tg.vs[v]['last_visit'][r]: " +str(g.vs[v]['last_visit'][r]) + "\tall last visit weigthing: " + str((1+vis*(step-g.vs[v]['last_visit'][r])/(len(g.es)*0.5*(instepsMax+instepsMin))))
    		
    		    #test
        	    #print "v "+str(v)+"\trat "+str(r)+"\trats[r]['poz'] "+str(rats[r]['poz'])+"\tneig[v] "+str(neig[v])+"\n"
            
            
        	    osszcucc+=neig[v]
    		sorskeze=osszcucc*random()
    		count=0
		#for v in neig.keys():
		    #print "test rat: "+ str(r)+ "v: "+ str(v) + "neig[v]: " + str(neig[v])
        
    		for v in neig.keys():
    		    count+=neig[v]
    		    if count>sorskeze:
            		go_to=v
            		break
            	    
            # check the stat comparing the other option:
    	    #if 0: ##### REMOVE THIS FOR TURN BACK	# len(downwards_choices) >1:
    		#del downwards_choices[go_to]
    		#print "downwards_choices: " + str(downwards_choices)
    		# only save stats if coming from the route, so the choices are symmetrical
    		#if rats[r]['prev_poz'] < rats[r]['poz']:
    		#    other = next(iter(downwards_choices))
    		    #print "other: " + str(other)
    		    #print "ratio_unvis[go_to]: " + str(ratio_unvis[go_to]) + "\t" + str(ratio_unvis[other])
    		#    if ratio_unvis[go_to] > ratio_unvis[other]:
    	    stat["towards_more_unvisited"]=1
    			#print "go_to>" 
    		#    elif ratio_unvis[go_to] < ratio_unvis[other]:
    	    stat["towards_less_unvisited"]=1
    			#print "go_to<"
    		    #print "reltime_lastvis[go_to]: " + str(reltime_lastvis[go_to]) + "\t" + str(reltime_lastvis[other])
    		    #if reltime_lastvis[go_to] < reltime_lastvis[other]:
    	    stat["towards_recent_last_visit"]=1
    			#print "go_to<"
    		    #elif reltime_lastvis[go_to] > reltime_lastvis[other]:
    	    stat["towards_long_ago_last_visit"]=1
    			#print "go_to>"
    		    #print "partners[go_to]: " + str(partners[go_to]) + "\t" + str(partners[other])
    		    #if partners[go_to] > 0 and partners[other]==0:
    	    stat["towards_partner"]=1
    			#print "go_to: partner"
    		    #elif partners[go_to] == 0 and partners[other]>0:
    	    stat["towards_no_partner"]=1
    			#print "go_to: no partner"
            
            	    
    	    # randomly decide how many steps will this be on the edge until reaching the node (within the range defined on the beginning of the script)
    	    rats[r]['steps_remaining']= choice(range(instepsMin,instepsMax))	#instepsMin + round((instepsMax-instepsMin)*random()))
	if go_to!=rats[r]['poz']:
    	    #print "r: " +str(r) + "\tprev: " + str(rats[r]['prev_poz']) + "\tpoz: " + str(rats[r]['poz']) + "\tgo_to: " + str(go_to)
    	    # register that this rat visited this edge
    	    eid = -1
    	    if rats[r]['prev_poz'] >=0:
    		eid = get_eid1(g, rats[r]['prev_poz'], rats[r]['poz'])
    		if eid>=0:
    		    g.es[eid]['here_are'].remove(r)
    		else:
    		    print "NOT found edge: "
	    
	    #eid = -1
    	    eid = get_eid1(g, rats[r]['poz'], go_to )
	    if eid>=0:
		g.es[eid]['here_are'].append(r)
    	    
    	    
    	    rats[r]['prev_poz']=rats[r]['poz']
    	    rats[r]['poz']=go_to
    	    
    	    # register that this rat visited this node
    	    if r not in g.vs[rats[r]['poz']]['visited_by']:
    		#only append the list if it is not already there
    		g.vs[rats[r]['poz']]['visited_by'].append(r) 
    	    #print "visited_by: " + str(g.vs[rats[r]['poz']]['visited_by'])
	    
	    # register the time of this visit
	    g.vs[rats[r]['poz']]['last_visit'][r] = step + rats[r]['steps_remaining']
	    #print "r " + str(r) + "\tlast visit" + str(g.vs[rats[r]['poz']]['last_visit'])
	    
    	    g.vs[rats[r]['prev_poz']]['here_are'].remove(r)
    	    g.vs[rats[r]['poz']]['here_are'].append(r) 
    
def step_rand_round(step):
    listofrats=range(n)
    while len(listofrats)>0:
        k=choice(listofrats)
        step_one(k, g, rats, goal, n, h, m, s, vis, step)
        listofrats.remove(k)

#def step_rand_one():
#    k=choice(range(n))
#    step_one(k, g, rats, goal, n, h, m, s)



#def step_round(g):
#    for r in range(n):
#        step_one(r, g, rats, goal, n, h, m, s)
#    #print g.vs["here_are"]    

'''
def step_parallel(n, rats, g, goal, h, m, s):
    go_to={x:-1 for x in range(n)}
    for r in range(n):
        neig={v:1 for v in g.neighbors(rats[r]['poz'])}
        neig[rats[r]['poz']]=1
        osszcucc=0
        if rats[r]['poz']==goal:
            go_to[r]=rats[r]['poz']
        else:
            for v in neig.keys():

                if v==rats[r]['prev_poz']:
                    neig[v]=h*s**len(g.vs[v]['here_are'])
                elif v==rats[r]['poz']:
                    neig[v]=m*s**(len(g.vs[v]['here_are'])-1)
                else:
                    neig[v]=s**len(g.vs[v]['here_are'])
                osszcucc+=neig[v]
            sorskeze=osszcucc*random()
            count=0
            for v in neig.keys():
                count+=neig[v]
                if count>sorskeze:
                    go_to[r]=v
                    break

    for r in range(n):
        if go_to[r]!=rats[r]['poz']:
            rats[r]['prev_poz']=rats[r]['poz']
            rats[r]['poz']=go_to[r]
            g.vs[rats[r]['prev_poz']]['here_are'].remove(r)
            g.vs[rats[r]['poz']]['here_are'].append(r)
    #print g.vs["here_are"]    
'''
'''
level_list=[4]
n_list=[8]
h_list=[0.5, 1, 1.5, 2]
m_list=[0.5, 1, 1.5, 2]
s_list=[0.2, 0.5, 1, 2, 5]

for level in level_list:
    for n in n_list:
        for h in h_list:
            for m in m_list:
                for s in s_list:
'''


out='n'+str(n)+'_l'+str(level)+'_h'+str(h)+'_m'+str(m)+'_s'+str(s)+'_vis'+str(vis)+'_toymodel'
with open(out+'.txt', "w") as myfile:
    imax=iter_num	#	1000
    arrives = []
    stats = []
    for i in range(imax):
        #Create a graph
        g=Graph.Tree(num_ver(level), 2)
        
        #Get the largest cluster, delete others
        cl = g.clusters()
        g = cl.giant()

        #Plot the track
        '''
        layout=g.layout("tree")
        plot(g, layout=layout)
        '''
        #g.write_edgelist('edgelist.txt')

        #Give the name numbers to the vertices
        g.vs["name"]=range(len(g.vs))
        g.vs["here_are"]=[[] for vert in range(len(g.vs))]
        g.vs["visited_by"]=[[] for vert in range(len(g.vs))]
        #for i in g.vs:
            #print i["name"]
        #for idx, v in enumerate(g.vs):
    	#    print str(v) + "\there_are\t" + str(v["here_are"])

        g.es["name"]=range(len(g.es))
        g.es["here_are"]=[[] for e in range(len(g.es))]
    	    
	
	# store for each node the time of the last visit by each rat
        g.vs["last_visit"]=[[0]*n for vert in range(len(g.vs))]
        #for v in enumerate(g.vs):
    	#    print str(v)	#v["last_visit"])
		
        start=g.vs[0]
        
        # create a list for storing occurences of choices
        stat={}
        stat["towards_more_unvisited"]=0
        stat["towards_less_unvisited"]=0
        stat["towards_recent_last_visit"]=0
        stat["towards_long_ago_last_visit"]=0
        stat["towards_partner"]=0
        stat["towards_no_partner"]=0
        
	
	# store from which nodes which endpoints are reachable by walking down the hierarchy
        g.vs["reachable_endpoints"]=[[] for vert in range(len(g.vs))]
	for vid in range(num_ver(level-1),num_ver(level)):
	    g.vs[vid]['reachable_endpoints']=[vid]
	for ilevel in range(1,level):
	    for vid in range(num_ver(level-ilevel-1),num_ver(level-ilevel)):
		#print "vid: " + str(vid) + "\tsuccessors" + str(g.successors(vid))
		for v in g.successors(vid):
		    if v>vid:
			#print str(v) + "g.vs[v]['reachable_endpoints'] " + str(g.vs[v]['reachable_endpoints'])
			g.vs[vid]['reachable_endpoints'] += ( g.vs[v]['reachable_endpoints'] )

        goal=choice(range(num_ver(level-1),num_ver(level)))

        start['here_are']=range(n)

        rats={t:{'poz':start.index, 'prev_poz':-1, 'steps_remaining':choice(range(0,instepsMax))} for t in range(n)}
	
	# file to write the trajectories
	if(write_traj):
	    fOUT = open('%s_%d.txt' %(out,i), 'w')
	    fOUT.write("#t\t{Now\tPrev\tSteps_remaining}\t...\n")
        step=0
        stepmax=10000
        while step<stepmax:
            if(write_traj):
        	#print out the trajectory for each rat
        	line = str(step)+"\t"
        	for r in range(n):
        	    line += str(rats[r]['poz'])+"\t"+str(rats[r]['prev_poz'])+"\t"+str(rats[r]['steps_remaining'])+"\t"
        	fOUT.write(line + "\n")
        	
	    # step
            #print step
            step+=1
            step_rand_round(step)
    	    #print "stats: " + "%d\t"* 6 %(stat["towards_more_unvisited"],
    	    #		  stat["towards_less_unvisited"],
    	    #		  stat["towards_recent_last_visit"],
    	    #		  stat["towards_long_ago_last_visit"],
    	    #		  stat["towards_partner"],
    	    #		  stat["towards_no_partner"])
            
            # if all arrived, end
            if len(g.vs[goal]['here_are'])==n:
                break
            #if step % (stepmax/10) ==0:
                #print 'actual step: ', step
        arrive = sorted(g.vs[goal]['last_visit'])
        arrives.append(arrive)
        #print str(arrives)
        out_line = "goal_last_visit: " + str(arrive)
        out_line += "\tstat:\t"
        out_line += "%d\t"* 6 %(stat["towards_more_unvisited"],
    			  stat["towards_less_unvisited"],
    			  stat["towards_recent_last_visit"],
    			  stat["towards_long_ago_last_visit"],
    			  stat["towards_partner"],
    			  stat["towards_no_partner"])
        if (stat["towards_more_unvisited"]+stat["towards_less_unvisited"])!=0 and (stat["towards_recent_last_visit"]+stat["towards_long_ago_last_visit"])!=0 and (stat["towards_partner"]+stat["towards_no_partner"])!=0:
    	    stat_ratio = [ 1.0*stat["towards_more_unvisited"]/(1.0*stat["towards_more_unvisited"]+1.0*stat["towards_less_unvisited"]), 
    		       1.0*stat["towards_less_unvisited"]/(1.0*stat["towards_more_unvisited"]+1.0*stat["towards_less_unvisited"]),
    		       1.0*stat["towards_recent_last_visit"]/(1.0*stat["towards_recent_last_visit"]+1.0*stat["towards_long_ago_last_visit"]),
    		       1.0*stat["towards_long_ago_last_visit"]/(1.0*stat["towards_recent_last_visit"]+1.0*stat["towards_long_ago_last_visit"]),
    		       1.0*stat["towards_partner"]/(1.0*stat["towards_partner"]+1.0*stat["towards_no_partner"]),
    		       1.0*stat["towards_no_partner"]/(1.0*stat["towards_partner"]+1.0*stat["towards_no_partner"]) ]
	    stats.append(stat_ratio)
        if(write_traj):
    	    fOUT.close()
        myfile.write(str(step)+"\t"+out_line+'\n')
        #if i%(10)==0:
        #    print 'i=', i

arrives_a = np.array(arrives)
#print str(np.mean(arrives_a, axis=0))

stats_a = np.array(stats)
#print str(np.mean(stats_a, axis=0))

print 'n '+str(n)+' l '+str(level)+' h '+str(h)+' m '+str(m)+' s '+str(s)+' vis '+str(vis)+' iter_num '+str(iter_num)+' arrives '+str(np.mean(arrives_a, axis=0))+' stats '+str(np.mean(stats_a, axis=0))


'''
gnuout=out+'.plt'
with open(gnuout, 'w') as plt:
    plt.write("#!/usr/bin/gnuplot\n"\
              +"binwidth=50\n"+"bin(x,width)=width*floor(x/width)\n"\
              +'plot \"'+str(out)+'.txt\" using (bin($1,binwidth)):(1.0) smooth freq with boxes')
    plt.write('\n pause -1 \"Press enter!\"')
    plt.write("\n"+"set term pngcairo transparent enhanced font \"helvetica,10\" \n\
set out \""+str(out)+".png\" \n\
replot")

from os import system
system('gnuplot '+str(gnuout))
'''


                        


