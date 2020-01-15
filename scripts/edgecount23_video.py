#!/usr/bin/python
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

from time import time
time1=time()

'''
THIS IS NOT PART OF THIS PROGRAM
import argparse
level=int(sys.argv[2])
ap = argparse.ArgumentParser()
#ap.add_argument("-t", "--template", required=True, help="Path to template image")
ap.add_argument("-i", "--image", required=True,
	help="Path to image where template will be matched")
args = vars(ap.parse_args())
'''


run=sys.argv[1]

def cosd(x):
    return cos(x*pi/180)
def sind(x):
    return sin(x*pi/180)


#Rotate the image and the sizes are changing to suits the image
def rotate(image, fi):
    if isfile(str(image)):
        imgr = cv2.imread(image)
    else:
        imgr=image
    rowsr,colsr = imgr.shape[:2]
    

    
    M = cv2.getRotationMatrix2D((0,0),0,1)
    imgr=cv2.warpAffine(imgr,M,(max(rowsr,colsr)*2,max(rowsr,colsr)*2))

    if fi>=0:
        M = np.float32([[1,0,0],[0,1,(colsr*sin(fi*pi/180))]])
        imgr=cv2.warpAffine(imgr,M,(max(rowsr,colsr)*2,max(rowsr,colsr)*2))

        #print int(colsr*cos(fi*pi/180)+rowsr*sin(fi*pi/180)), int(colsr*sin(fi*pi/180)+rowsr*cos(fi*pi/180))

        #cv2.imwrite('rot0.jpg',imgr)

        M = cv2.getRotationMatrix2D((0,colsr*sin(fi*pi/180)),fi,1)
        imgr = cv2.warpAffine(imgr,M,(int(colsr*cos(fi*pi/180)+rowsr*sin(fi*pi/180)),\
                                    int(colsr*sin(fi*pi/180)+rowsr*cos(fi*pi/180))))

        #cv2.imwrite('rot.jpg',imgr)
        return imgr
    else:
        M = np.float32([[1,0,(rowsr*sin(abs(fi*pi/180)))],[0,1,0]])
        imgr=cv2.warpAffine(imgr,M,(max(rowsr,colsr)*2,max(rowsr,colsr)*2))
        M = cv2.getRotationMatrix2D((rowsr*sin(abs(fi*pi/180)),0),fi,1)
        imgr = cv2.warpAffine(imgr,M,(int(colsr*cos(abs(fi)*pi/180)+rowsr*sin(abs(fi)*pi/180)),\
                                    int(colsr*sin(abs(fi)*pi/180)+rowsr*cos(abs(fi)*pi/180))))
        return imgr



#Expand image to double size. Image is in the centre.
def expand(image):
    if isfile(str(image)):
        imgr = cv2.imread(image)
    else:
        imgr=image
    rowsr,colsr = imgr.shape[:2]
    
    M = cv2.getRotationMatrix2D((0,0),0,1)
    imgr=cv2.warpAffine(imgr,M,(colsr*2,rowsr*2))
    M = np.float32([[1,0,colsr/2],[0,1,rowsr/2]])
    imgr=cv2.warpAffine(imgr,M,(colsr*2,rowsr*2))
    #cv2.imwrite('exp0.jpg',imgr)
    return imgr




#Import image to be analyzed, and find the rotation angle with template
best={} #angle, correlation value, location
top_left=[0,0]

original=[cv2.imread('./jpg_10000/'+run+'.ts_00010000.jpg')] #original
illesztett=original[0].copy()
mask=cv2.imread('mask2.jpg',0)
template = cv2.imread('temp4.jpg',0)
template2=template.copy()
w, h = template.shape[::-1]
ow= original[0].shape[1]
oh= original[0].shape[0]

def match_img(imgfile):
    img = cv2.imread(imgfile,0)
    original[0]=cv2.imread(imgfile) #original

    img = cv2.medianBlur(img,15)
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                                cv2.THRESH_BINARY,11,2)
    img=cv2.bitwise_not(img)
    
    img=cv2.bitwise_and(img,mask)
    #cv2.imwrite('prob1.jpg',img)
    img2 = img.copy()
    
    #print "template shape: width ", w, "height ", h

    meth='cv2.TM_CCOEFF_NORMED'
    angles=np.linspace(-0.05,0.05,3)
    #angles=range(0,1)

     
    for fi in angles:
        print "METHOD: ", meth, "FI: ", fi
        img = expand(img2.copy())
        template=template2.copy()
        #cv2.imwrite('expanged_img.jpg',img)
        print "expanded image shape: width ", img.shape[1], "height ", img.shape[0]
        
        method = eval(meth)

        template=rotate(template, fi)
        #cv2.imwrite(meth+'rotated'+str(fi)+'.jpg',template)
        
        # Apply template Matching
        res = cv2.matchTemplate(img,template,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        print "min_val, max_val, min_loc, max_loc", min_val, max_val, min_loc, max_loc

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:        
            if (len(best)==0) or (min_val<best['corr']):
                best['ang']=fi
                ffi=fi
                best['corr']=min_val
                best['loc']=min_loc
        else:        
            if (len(best)==0) or (max_val>best['corr']):
                best['ang']=fi
                ffi=fi
                best['corr']=max_val
                best['loc']=max_loc


    top_left[0],top_left[1] = best['loc'][0],best['loc'][1]
    #ffi=best['ang']
    print best
    
    #bottom_right = (top_left[0] + w, top_left[1] + h)
    #cv2.rectangle(img,top_left, bottom_right, 255, 2)
    illesztett=original[0].copy()
    ow= original[0].shape[1]
    oh= original[0].shape[0]

match_img('./jpg_10000/'+run+'.ts_00010000.jpg')

with open('./runs/'+run+'/'+run+'_image_match_params.txt','w') as imparfile:
    imparfile.write('ang\ttop_left[0]\ttop_left[1]\tcorr\n')
    imparfile.write(str(best['ang'])+'\t'+str(best['loc'][0])+'\t'+str(best['loc'][1])+'\t'+str(best['corr']))
#cv2.imwrite('ell1.jpg',original[0])
#print best


g=Graph()
f=[ [ [796,312],[904,362] ],   [[904,362],[1014,304]], [[834,458],[976,460]]   ]
g.add_vertices(52)
with open('edges.txt', 'r') as efil:
    for esor in efil:
        node1=int(esor.split()[0])
        node2=int(esor.split()[1])
        g.add_edges([(node1,node2)])

illesztett=original[0].copy()
with open('nodes_2.txt', 'r') as nfil:
    for nsor in nfil:
        ffi=best['ang']
        node=int(nsor.split()[0])
        font = cv2.FONT_HERSHEY_SIMPLEX
        xx=int(nsor.split()[1])
        yy=int(nsor.split()[2])
        if ffi>=0:
            x=int(xx*cosd(ffi)+yy*sind(ffi)+top_left[0]-ow*0.5)
            y=int((-sind(ffi))*xx+cosd(ffi)*yy+top_left[1]+w*sind(ffi)-oh*0.5)
        else:
            x=int(xx*cosd(ffi)+yy*sind(ffi)+top_left[0]+h*sind(abs(ffi))-ow*0.5)
            y=int((-sind(ffi))*xx+cosd(ffi)*yy+top_left[1]-oh*0.5)
                           
        g.vs[node]['x']=x
        g.vs[node]['y']=y
        #g.vs[node]['gx']=x/1920.0
        #g.vs[node]['gy']=y/1080.0
        cv2.putText(illesztett,str(node),(x,y), font, 1,(0,0,0),2)

for index, edge in enumerate(g.get_edgelist()):
    x1=g.vs[edge[0]]['x']
    y1=g.vs[edge[0]]['y']
    x2=g.vs[edge[1]]['x']
    y2=g.vs[edge[1]]['y']
    cv2.line(illesztett,(x1,y1),(x2,y2),(0,0,255),2)
    g.es[index]['l']=sqrt((x2-x1)**2+(y2-y1)**2)
    #print g.es[index]['l']
#cv2.imwrite('ill'+str(ffi)+'.jpg',illesztett)


def refresh_node_poz():
    ffi=best['ang']
    illesztett=original[0].copy()
    with open('nodes_2.txt', 'r') as nfil:
        for nsor in nfil:
            
            node=int(nsor.split()[0])
            font = cv2.FONT_HERSHEY_SIMPLEX
            xx=int(nsor.split()[1])
            yy=int(nsor.split()[2])
            if ffi>=0:
                x=int(xx*cosd(ffi)+yy*sind(ffi)+top_left[0]-ow*0.5)
                y=int((-sind(ffi))*xx+cosd(ffi)*yy+top_left[1]+w*sind(ffi)-oh*0.5)
            else:
                x=int(xx*cosd(ffi)+yy*sind(ffi)+top_left[0]+h*sind(abs(ffi))-ow*0.5)
                y=int((-sind(ffi))*xx+cosd(ffi)*yy+top_left[1]-oh*0.5)
                               
            g.vs[node]['x']=x
            g.vs[node]['y']=y
            #g.vs[node]['gx']=x/1920.0
            #g.vs[node]['gy']=y/1080.0
            cv2.putText(illesztett,str(node),(x,y), font, 1,(0,0,0),2)

    for index, edge in enumerate(g.get_edgelist()):
        x1=g.vs[edge[0]]['x']
        y1=g.vs[edge[0]]['y']
        x2=g.vs[edge[1]]['x']
        y2=g.vs[edge[1]]['y']
        cv2.line(illesztett,(x1,y1),(x2,y2),(0,0,255),2)
        g.es[index]['l']=sqrt((x2-x1)**2+(y2-y1)**2)
        #print g.es[index]['l']
    for wall in f:
        for wp in wall:
            xx=float(wp[0])
            yy=float(wp[1])
            if ffi>=0:
                x=int(xx*cosd(ffi)+yy*sind(ffi)+top_left[0]-ow*0.5)
                y=int((-sind(ffi))*xx+cosd(ffi)*yy+top_left[1]+w*sind(ffi)-oh*0.5)
            else:
                x=int(xx*cosd(ffi)+yy*sind(ffi)+top_left[0]+h*sind(abs(ffi))-ow*0.5)
                y=int((-sind(ffi))*xx+cosd(ffi)*yy+top_left[1]-oh*0.5)
            wp[0]=x
            wp[1]=y
        cv2.line(illesztett,(wall[0][0],wall[0][1]),(wall[1][0],wall[1][1]),(0,255,0),2)

    cv2.imwrite('./matched_images/'+run+'_ill'+str(ffi)+'.jpg',illesztett)

#sys.exit()
refresh_node_poz()

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

targets_edges = { }
for t in range(1,17):
    targets_edges[t] = g.get_eid( targets_nodes[t][0] , targets_nodes[t][1] )

targets_prev_edges = { }
for t in range(1,17):
    targets_prev_edges[t] = g.get_eid( targets_nodes[t][1] , targets_nodes[t][2] )

edges_targets = { }
prev_edges_targets = { }

for t, e in targets_edges.iteritems():
    edges_targets[e] = t

for t, pe in targets_prev_edges.iteritems():
    prev_edges_targets[pe] = t

'''
def pozition(x,y,tryedges):
    poz={'eid':-1, 'diff':-1, 'd1':-1, 'dnormed':-1 }
    for ed in tryedges:
        d1t=sqrt((x-g.vs[min(g.get_edgelist()[ed])]['x'])**2+(y-g.vs[min(g.get_edgelist()[ed])]['y'])**2)
        difft=-g.es[ed]['l']+d1t+sqrt((x-g.vs[max(g.get_edgelist()[ed])]['x'])**2+(y-g.vs[max(g.get_edgelist()[ed])]['y'])**2)
        if (poz['eid']==-1) or difft<poz['diff']:
            poz['eid']=ed
            poz['diff']=difft
            poz['d1']=d1t
    if poz['eid'] not in halfs:
        poz['dnormed']=poz['d1']/g.es[ed]['l']
    else:
        poz['dnormed']=poz['d1']/(g.es[ed]['l']*2)
    return poz
'''

'''
a=pozition(422,770,range(0,48))
print a
print g.get_edgelist()[a['eid']]
print g.vs[min(g.get_edgelist()[ a['eid'] ])].index    
'''

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

#print lr_vect(590,442,904,418,406,304)

def linelen(x1,y1,x2,y2):
    return sqrt((x2-x1)**2+(y2-y1)**2)

def linelensq(x1,y1,x2,y2):
    return (x2-x1)**2+(y2-y1)**2

#print g.neighbors(11)

for node in g.vs:
    node['numneigh']=len(g.neighbors(node.index))

def cross_lines(x11,y11,x12,y12,x21,y21,x22,y22):
    x11=float(x11)
    x12=float(x12)
    x21=float(x21)
    x22=float(x22)
    y11=float(y11)
    y12=float(y12)
    y21=float(y21)
    y22=float(y22)
    res={}
    if (x12-x11)!=0:
        m1=(y12-y11)/(x12-x11)
    else:
        m1=float('inf')
        
    if (x22-x21)!=0:
        m2=(y22-y21)/(x22-x21)
    else:
        m2=float('inf')
        
    if m1==m2:
        res['x']=float('inf')
        res['y']=float('inf')
    elif m1==float('inf'):
        res['x']=x11
        res['y']=y21+m2*(res['x']-x21)
    elif m2==float('inf'):
        res['x']=x21
        res['y']=y11+m1*(res['x']-x11)     
    else:     
        res['x']=(y21-x21*m2-y11+x11*m1)/(m1-m2)
                  
        res['y']=(y11+m1*(res['x']-x11))
        #res['x']=int(res['x'])

    return res

def cross_linesm(x1,y1,m1,x2,y2,m2):
    res={}
    x1=float(x1)
    y1=float(y1)
    m1=float(m1)
    x2=float(x2)
    y2=float(y2)
    m2=float(m2)
   
    if m1==m2:
        res['x']=float('inf')
        res['y']=float('inf')
    elif m1==float('inf'):
        res['x']=x1
        res['y']=y2+m2*(res['x']-x2)
    elif m2==float('inf'):
        res['x']=x2
        res['y']=y1+m1*(res['x']-x1)     
    else:     
        res['x']=((y2-x2*m2-y1+x1*m1)/(m1-m2))
                  
        res['y']=(y1+m1*(res['x']-x1))

    return res


metszet=cross_lines(0,0,500,500,0,500,500,0)

#print metszet

def perptomid(x1,y1,x2,y2):
    x1,y1,x2,y2=float(x1),float(y1),float(x2),float(y2)
    res={}
    res['x']=(x1+x2)/2
    res['y']=(y1+y2)/2
    if (y1-y2)==0:
        m=float('inf')
    else:
        m=(x2-x1)/(y1-y2)
    res['m']=m
    #res['x'],res['y']=int(res['x']),int(res['y'])
    return res

#print perptomid(0,0,500,500)
def make_projectpoints():
    for s in g.vs:
        neighs=g.neighbors(s.index)
        s['alfa']=g.vs[neighs[0]]
        alfa=s['alfa']
        if len(neighs)>1:
            s['beta']=g.vs[neighs[1]]
            beta=s['beta']
            if len(neighs)>2:
                s['gamma']=g.vs[neighs[2]]
                gamma=s['gamma']
        if len(neighs)==3 and s.index!=0:
            s['projectpoints']=[{'leftof':[s['x'],s['y'],beta['x'],beta['y']],\
                                 'rightof':[s['x'],s['y'],gamma['x'],gamma['y']]},\
                                {'leftof':[s['x'],s['y'],gamma['x'],gamma['y']],\
                                 'rightof':[s['x'],s['y'],alfa['x'],alfa['y']]},\
                                {'leftof':[s['x'],s['y'],alfa['x'],alfa['y']],\
                                 'rightof':[s['x'],s['y'],beta['x'],beta['y']]}]
            x1,y1,m1,x2,y2,m2,x3,y3,m3=perptomid(s['x'],s['y'],alfa['x'],alfa['y'])['x'],\
                               perptomid(s['x'],s['y'],alfa['x'],alfa['y'])['y'],\
                               perptomid(s['x'],s['y'],alfa['x'],alfa['y'])['m'],\
                               perptomid(s['x'],s['y'],beta['x'],beta['y'])['x'],\
                               perptomid(s['x'],s['y'],beta['x'],beta['y'])['y'],\
                               perptomid(s['x'],s['y'],beta['x'],beta['y'])['m'],\
                               perptomid(s['x'],s['y'],gamma['x'],gamma['y'])['x'],\
                               perptomid(s['x'],s['y'],gamma['x'],gamma['y'])['y'],\
                               perptomid(s['x'],s['y'],gamma['x'],gamma['y'])['m']
                               
            s['projectpoints'][0]['x']=cross_linesm(x2,y2,m2,x3,y3,m3)['x']
            s['projectpoints'][0]['y']=cross_linesm(x2,y2,m2,x3,y3,m3)['y']
            s['projectpoints'][1]['x']=cross_linesm(x3,y3,m3,x1,y1,m1)['x']
            s['projectpoints'][1]['y']=cross_linesm(x3,y3,m3,x1,y1,m1)['y']
            s['projectpoints'][2]['x']=cross_linesm(x1,y1,m1,x2,y2,m2)['x']
            s['projectpoints'][2]['y']=cross_linesm(x1,y1,m1,x2,y2,m2)['y']

        elif s.index==0:
            fx=(2*s['x']-gamma['x'])
            fy=(2*s['y']-gamma['y'])
            s['projectpoints']=[{'leftof':[s['x'],s['y'],beta['x'],beta['y']],\
                                 'rightof':[s['x'],s['y'],gamma['x'],gamma['y']]},\
                                {'leftof':[s['x'],s['y'],gamma['x'],gamma['y']],\
                                 'rightof':[s['x'],s['y'],alfa['x'],alfa['y']]},\
                                {'leftof':[s['x'],s['y'],alfa['x'],alfa['y']],\
                                 'rightof':[s['x'],s['y'],fx,fy]},\
                                {'leftof':[s['x'],s['y'],fx,fy],\
                                 'rightof':[s['x'],s['y'],beta['x'],beta['y']]},\
                                ]
            
            x1,y1,m1,x2,y2,m2,x3,y3,m3,x4,y4,m4=perptomid(s['x'],s['y'],alfa['x'],alfa['y'])['x'],\
                               perptomid(s['x'],s['y'],alfa['x'],alfa['y'])['y'],\
                               perptomid(s['x'],s['y'],alfa['x'],alfa['y'])['m'],\
                               perptomid(s['x'],s['y'],beta['x'],beta['y'])['x'],\
                               perptomid(s['x'],s['y'],beta['x'],beta['y'])['y'],\
                               perptomid(s['x'],s['y'],beta['x'],beta['y'])['m'],\
                               perptomid(s['x'],s['y'],gamma['x'],gamma['y'])['x'],\
                               perptomid(s['x'],s['y'],gamma['x'],gamma['y'])['y'],\
                               perptomid(s['x'],s['y'],gamma['x'],gamma['y'])['m'],\
                               perptomid(s['x'],s['y'],fx,fy)['x'],\
                               perptomid(s['x'],s['y'],fx,fy)['y'],\
                               perptomid(s['x'],s['y'],fx,fy)['m']
                               
            s['projectpoints'][0]['x']=cross_linesm(x2,y2,m2,x3,y3,m3)['x']
            s['projectpoints'][0]['y']=cross_linesm(x2,y2,m2,x3,y3,m3)['y']
            s['projectpoints'][1]['x']=cross_linesm(x3,y3,m3,x1,y1,m1)['x']
            s['projectpoints'][1]['y']=cross_linesm(x3,y3,m3,x1,y1,m1)['y']
            s['projectpoints'][2]['x']=cross_linesm(x1,y1,m1,x4,y4,m4)['x']
            s['projectpoints'][2]['y']=cross_linesm(x1,y1,m1,x4,y4,m4)['y']
            s['projectpoints'][3]['x']=cross_linesm(x2,y2,m2,x4,y4,m4)['x']
            s['projectpoints'][3]['y']=cross_linesm(x2,y2,m2,x4,y4,m4)['y']            
           
            
        elif len(neighs)==2:
            s['projectpoints']=[{}]
            x1,y1,m1,x2,y2,m2=perptomid(s['x'],s['y'],alfa['x'],alfa['y'])['x'],\
                               perptomid(s['x'],s['y'],alfa['x'],alfa['y'])['y'],\
                               perptomid(s['x'],s['y'],alfa['x'],alfa['y'])['m'],\
                               perptomid(s['x'],s['y'],beta['x'],beta['y'])['x'],\
                               perptomid(s['x'],s['y'],beta['x'],beta['y'])['y'],\
                               perptomid(s['x'],s['y'],beta['x'],beta['y'])['m']
            s['projectpoints'][0]['x']=cross_linesm(x1,y1,m1,x2,y2,m2)['x']
            s['projectpoints'][0]['y']=cross_linesm(x1,y1,m1,x2,y2,m2)['y']

        else: #len(neighs)==1
            s['projectpoints']=alfa['projectpoints']
    g.vs[49]['projectpoints'][0]['x']=2*g.vs[49]['x']-g.vs[0]['x']
    g.vs[49]['projectpoints'][0]['y']=2*g.vs[49]['y']-g.vs[0]['y']


make_projectpoints()

def invtr(x,y):
    ffi=best['ang']
    if ffi>=0:
        xt=cosd(ffi)*(x+ow*0.5-top_left[0])-sind(ffi)*(y-(top_left[1]+w*sind(ffi)-oh*0.5))
        yt=sind(ffi)*(x+ow*0.5-top_left[0])+cosd(ffi)*(y-(top_left[1]+w*sind(ffi)-oh*0.5))
    else:
        xt=cosd(ffi)*(x-(top_left[0]-h*sind(ffi)-ow*0.5))-sind(ffi)*(y+oh*0.5-top_left[1])
        yt=sind(ffi)*(x-(top_left[0]-h*sind(ffi)-ow*0.5))+cosd(ffi)*(y+oh*0.5-top_left[1])
    return [xt, yt]

part1=[5,19,9,18,34,35]
part2=[35,9,5,19,10,20,36,3,17,21,6,11]
part3=[36,20,21,37,11,22,38,6]
part4=[9,5,3,17,33,6,1]
part5=[51,49,50,0,1,2,40,41]
part6=[48,32,16,8,4,2,7]
part7=[3,6,11,12,23,24,40,41,25,13,26,7,4,14]
part8=[39,23,12,13,26,42,27,14]
part9=[32,16,8,31,47,46,30]
part10=[4,8,16,15,30,46,29,45,28]
part11=[43,27,14,7,28,44,45,29]

def projector(x,y):#,prevnode,prevedge,prev_prpont_id):
    xit, yit= invtr(x,y)
    if xit<516:
        if yit<242:
            trynodes=part1
        elif yit<634:
            trynodes=part2
        else:
            trynodes=part3
    elif xit<1326:
        if yit<522:
            if xit<746:
                trynodes=part4
            elif xit<1076:
                trynodes=part5
            else:
                trynodes=part6
        elif yit<800:
            trynodes=part7
        else:
            trynodes=part8
    else:
        if yit<264:
            trynodes=part9
        elif yit<660:
            trynodes=part10
        else:
            trynodes=part11

    x=float(x)
    y=float(y)
    node=-1
    dsq_node_patek=-1
    for n in trynodes:
        dsq_try=linelensq(g.vs[n]['x'],g.vs[n]['y'],x,y)
        if node==-1 or dsq_try<dsq_node_patek:
            dsq_node_patek=dsq_try
            node=n
    if node==50 or node==51:
        node=49
    if (node==1) or (node==2):
        if (lr_vect(x,y,f[0][0][0],f[0][0][1],f[0][1][0],f[0][1][1])=='left') and (lr_vect(x,y,f[1][0][0],f[1][0][1],f[1][1][0],f[1][1][1])=='left'):
            node=49
    elif (node==40):  
        if (lr_vect(x,y,f[2][0][0],f[2][0][1],f[2][1][0],f[2][1][1])=='left'):
            if (linelensq(g.vs[0]['x'],g.vs[0]['y'],x,y)<linelensq(g.vs[1]['x'],g.vs[1]['y'],x,y)):
                node=0
            else:
                node=1
    elif (node==41):
        if (lr_vect(x,y,f[2][0][0],f[2][0][1],f[2][1][0],f[2][1][1])=='left'):
            if (linelensq(g.vs[0]['x'],g.vs[0]['y'],x,y)<linelensq(g.vs[2]['x'],g.vs[2]['y'],x,y)):
                node=0
            else:
                node=2

    s=g.vs[node]
    numneigh=s['numneigh']
    prpont_id=-1
    n2=g.vs[0]
    if numneigh==3 and s.index!=0:
        for i in range(3):
            checkleft=s['projectpoints'][i]['leftof']
            checkright=s['projectpoints'][i]['rightof']
            if lr_vect(x,y,checkleft[0],checkleft[1],checkleft[2],checkleft[3])=='left' \
               and lr_vect(x,y,checkright[0],checkright[1],checkright[2],checkright[3])=='right':
                prpont_id=i
                break
        
        if prpont_id==0:
            if lr_vect(x,y,s['x'],s['y'],s['projectpoints'][0]['x'],s['projectpoints'][0]['y'])=='left':
                n2=s['gamma']
            else:
                n2=s['beta']

        elif prpont_id==1:
            if lr_vect(x,y,s['x'],s['y'],s['projectpoints'][1]['x'],s['projectpoints'][1]['y'])=='left':
                n2=s['alfa']
            else:
                n2=s['gamma']

        else:
            if lr_vect(x,y,s['x'],s['y'],s['projectpoints'][2]['x'],s['projectpoints'][2]['y'])=='left':
                n2=s['beta']
            else:
                n2=s['alfa']

    elif s.index==0:
        for i in range(4):
            checkleft=s['projectpoints'][i]['leftof']
            checkright=s['projectpoints'][i]['rightof']
            if lr_vect(x,y,checkleft[0],checkleft[1],checkleft[2],checkleft[3])=='left' \
               and lr_vect(x,y,checkright[0],checkright[1],checkright[2],checkright[3])=='right':
                prpont_id=i
                break

        if prpont_id==0:
            if lr_vect(x,y,s['x'],s['y'],s['projectpoints'][0]['x'],s['projectpoints'][0]['y'])=='left':
                n2=g.vs[49]
            else:
                n2=g.vs[2]
        elif prpont_id==1:
            if lr_vect(x,y,s['x'],s['y'],s['projectpoints'][1]['x'],s['projectpoints'][1]['y'])=='left':
                n2=g.vs[1]
            else:
                n2=g.vs[49]        
        elif prpont_id==2:
            if lr_vect(x,y,s['x'],s['y'],s['projectpoints'][2]['x'],s['projectpoints'][2]['y'])=='left':
                n2=g.vs[49]
            else:
                n2=g.vs[1]
        elif prpont_id==3:
            if lr_vect(x,y,s['x'],s['y'],s['projectpoints'][3]['x'],s['projectpoints'][3]['y'])=='left':
                n2=g.vs[2]
            else:
                n2=g.vs[49]   

            
    elif numneigh==2:
        prpont_id=0
        if node in [17,19,20,22,23,25,2,31,28]:
            if lr_vect(x,y,s['x'],s['y'],s['projectpoints'][0]['x'],s['projectpoints'][0]['y'])=='right':
                n2=s['alfa']
            else:
                n2=s['beta']
        else: 
            if lr_vect(x,y,s['x'],s['y'],s['projectpoints'][0]['x'],s['projectpoints'][0]['y'])=='left':
                n2=s['alfa']
            else:
                n2=s['beta']
            

    else: #numneigh==1:
        prpont_id=0
        n2=s['alfa']


    res=cross_lines(s['x'],s['y'],n2['x'],n2['y'],s['projectpoints'][prpont_id]['x'],s['projectpoints'][prpont_id]['y'],x,y)
    res['edge']=g.get_eid(node,n2.index)
    res['d']=linelen(res['x'],res['y'],g.vs[min(s.index,n2.index)]['x'],g.vs[min(s.index,n2.index)]['y'])
    res['dnorm']=res['d']/g.es[res['edge']]['l']
    if res['edge'] in egeszek:
        res['dnormgr'] = res['dnorm']
    else:
        res['dnormgr'] = res['dnorm']*0.5
   
    res['n1']=node
    res['n2']=n2.index
    res['ox']=x
    res['oy']=y
    #res['gx']=res['x']/1920.0
    #res['gy']=res['y']/1080.0
    #print "prpont_id", prpont_id
    #print g.vs[node]['projectpoints'][prpont_id]['x'],g.vs[node]['projectpoints'][prpont_id]['y'] 
    return res

'''
with open('viz_edges.txt', 'w') as vize:
    for index, edge in enumerate(g.get_edgelist()):
        gx1=g.vs[edge[0]]['x']
        gy1=g.vs[edge[0]]['y']
        gx2=g.vs[edge[1]]['x']
        gy2=g.vs[edge[1]]['y']
        dgx=gx2-gx1
        dgy=gy2-gy1
        vize.write(str(gx1)+'\t'+str(gy1)+'\t'+str(dgx)+'\t'+str(dgy)+'\n')
'''

rat_poz={}
rat_epop={}
reach_target={}
reach_target_prevedge={}
in_chamber=[]
#two_minutes_started=False

def check_in_chamber(color):
    if ((rat_poz[color]['edge']==g.get_eid(0,49)) and (lr_vect(rat_poz[color]['ox'],rat_poz[color]['oy'],f[0][0][0],f[0][0][1],f[0][1][0],f[0][1][1])=='left') and (lr_vect(rat_poz[color]['ox'],rat_poz[color]['oy'],f[1][0][0],f[1][0][1],f[1][1][0],f[1][1][1])=='left'))\
       or (rat_poz[color]['edge']==g.get_eid(49,50))\
       or (rat_poz[color]['edge']==g.get_eid(49,51)):
        if color not in in_chamber:
            in_chamber.append(color)
        return True
    else:
        if color in in_chamber:
            in_chamber.remove(color)
        return False



system('mkdir ./runs/'+run)

barcodes='./runs/'+run+'/'+run+'_gauss'

'''
for file in os.listdir('../../vasarhelyig/ratmaze/ratmaze_full_run__trajognize/done/ratmaze_full_run__trajognize__'+run+'.ts/OUT/'):
    if fnmatch.fnmatch(file, '*barcodes'):
        barcodes= file
        break
barcodes='../../vasarhelyig/ratmaze/ratmaze_full_run__trajognize/done/ratmaze_full_run__trajognize__'+run+'.ts/OUT/'+barcodes
'''

target = int(run.split('target')[1])
target_node = targets_nodes[target][0]
target_edge = targets_edges[target]
target_prev_edge = targets_prev_edges[target]

osszpatek=1

if run[0:5] == 'group':
    runtype = 'group'
    groupname = run.split('_target', 1)[0][-3:]
    gender=groupname[0]
    osszpatek=8

elif (run[0:5]=='learn') and (len(run.split('_')[2])==3):
    runtype = 'learnsg'
    groupname = run.split('_')[2]
    gender=groupname[0]
    osszpatek=8

elif (run[0:5]=='learn') and (len(run.split('_')[2])==4):
    runtype = 'learns'
    ratname = run.split('_')[2]
    gender=ratname[3]

elif run[0:6]=='single':
    runtype = 'single'
    ratname = run.split('_')[2]
    gender=ratname[3]

M12=['ORBM','OGBM','OBGM','GRBM','GRPM','GPBM','BRPM','BGPM']
M34=['ORPM','OBPM','OPGM','OPBM','GOBM','GOPM','GBPM','BOPM']
F12=['ROGF','ROBF','RGOF','RGBF','RGPF','ORBF','OGPF','GRBF']
F34=['ROPF','RBOF','RBGF','RBPF','RPOF','RPGF','RPBF','ORGF']


if osszpatek==1:
    rat_epop[ratname]=[0, [0 for edge in g.es()] ]
    reach_target[ratname] = -1
    reach_target_prevedge[ratname] = -1

else:
    for p, ratname in enumerate(eval(groupname)):
        rat_epop[ratname]= [ p,   [0 for edge in g.es()] ]
        reach_target[ratname] = -1
        reach_target_prevedge[ratname] = -1


projfile=open('./runs/'+run+'/'+run+'_proj','w')


with open(barcodes, 'r') as data:
    
    osszframe=0
    runstart_frame=-1
    with open('summarycoding14list.csv', 'r') as runstartfile:
        for line in runstartfile:
            lin = line.split('\t')
            if (lin[0] == run) and (lin[1] == "start"):
                runstart_frame = 1500*int(lin[2]) + 25*int(lin[3])
    print 'runstart frame: ', runstart_frame
    for ind, line in enumerate(data):
        #if ind==0:
        #    osszpatek=int(line.split()[4])
        if ind==1:
            osszframe=int(line.split()[4])
        if ind<13:
            projfile.write(line)
        elif ind==13:
            projfile.write('# fix width format: framenum barcodenum {ID centerx_original centery_original centerx_proj centery_proj edge d dnorm dnormgr n1 n2 xWorld yWorld orientation mFix} {...\n\n')
        
        #if ind<int(sys.argv[2]):
        #    continue
        if ind==100+runstart_frame:
            break
        #print ind
        if ind < 15+runstart_frame:
            continue
        sor=line.split()
        framenum=int(sor[0])
        pateknum=int(sor[1])
        in_chamber=[]

        projfile.write(sor[0]+'\t'+sor[1])

      


        with open('rats.txt', 'w') as rats:
            for p in range(pateknum):
                color=sor[2+p*7]+gender
                if color not in rat_epop.keys():
                    continue
                projfile.write('\t'+color)
                patx=float(sor[3+p*7])
                projfile.write('\t'+sor[3+p*7])
                #print patx
                paty=float(sor[4+p*7])
                projfile.write('\t'+sor[4+p*7])
                #print paty
                rat_poz[color]=projector(patx,paty) #,range(52)
                #if (runstart_frame==-1):
                #    check_in_chamber(color)
                if (rat_poz[color]['edge']==target_prev_edge) and (reach_target_prevedge[color]==-1):
                    reach_target_prevedge[color]=framenum-runstart_frame
                elif (rat_poz[color]['edge']==target_edge) and (reach_target[color]==-1):
                    reach_target[color]=framenum-runstart_frame
                         
                rats.write(str(rat_poz[color]['x'])+'\t'+str(rat_poz[color]['y'])+'\t'+str(rat_epop[color][0])+'\n')
                projfile.write('\t'+str(rat_poz[color]['x'])+'\t'+str(rat_poz[color]['y'])+'\t'+str(rat_poz[color]['edge'])+'\t'+str(rat_poz[color]['d'])+'\t'+str(rat_poz[color]['dnorm'])+'\t'+str(rat_poz[color]['dnormgr'])+'\t'+str(rat_poz[color]['n1'])+'\t'+str(rat_poz[color]['n2']))
                projfile.write('\t'+sor[5+p*7]+'\t'+sor[6+p*7]+'\t'+sor[7+p*7]+'\t'+sor[8+p*7])
                if  framenum > runstart_frame:
                    rat_epop[color][1][rat_poz[color]['edge']]+=1
            projfile.write('\n')
        '''
        if (two_minutes_started==False) and (osszpatek==len(in_chamber)):
            two_minutes_started=True
        
        if (runstart_frame==-1) and (pateknum>len(in_chamber)) and (two_minutes_started==True):
            for p in range(pateknum):
                color=sor[2+p*7]+gender
                if color not in rat_epop.keys():
                    continue
                if (rat_poz[color]['edge']==g.get_eid(0,1)) or (rat_poz[color]['edge']==g.get_eid(0,2)):
                    runstart_frame=framenum
        '''
        system('gnuplot -e "outfile=\'./runs/'+run+'/'+str(10000000+framenum)+'\'; pateks='+str(osszpatek)+'" faplotg4.4.plt')
        if (str(framenum)[-3:] == "000"):
            print framenum, '/', osszframe
system('mencoder "mf://./runs/'+run+'/10*.jpg" -mf fps=25 -o ./runs/'+run+'/'+run+'.mpg -ovc lavc -lavcopts vcodec=mpeg4')
system('rm ./runs/'+run+'/10*.jpg')

projfile.close()

index_color={}
for i in range(osszpatek):
    for color in rat_epop.keys():
        if rat_epop[color][0]==i:
            index_color[i]=color


with open('./runs/'+run+'/'+run+'_rat_epop.txt', 'w') as epp:
    epp.write('#eid \t x0 \t y0 \t dx \t dy \t ')
    for p in range(osszpatek):
        epp.write(index_color[p]+'\t')
    epp.write('sum \n')
    for index, edge in enumerate(g.get_edgelist()):
        summa=0
        gx1=g.vs[edge[0]]['x']
        gy1=g.vs[edge[0]]['y']
        gx2=g.vs[edge[1]]['x']
        gy2=g.vs[edge[1]]['y']
        dgx=gx2-gx1
        dgy=gy2-gy1
        epp.write(str(index)+'\t '+str(gx1)+' \t '+str(gy1)+' \t '+str(dgx)+' \t '+str(dgy)+' \t')
        for p in range(osszpatek):
            ez=rat_epop[index_color[p]][1][index]
            summa+=ez
            epp.write(str(ez)+'\t')
        epp.write(str(summa)+'\n')


 
'''
with open(barcodes+'_rat_epop.txt', 'w') as epp:
    for key, value in rat_epop.iteritems():
        epp.write(str(value[0])+'\t'+key+'\t')
        for edge in range(len(g.es())):
            epp.write(str(value[1][edge])+'\t')
        epp.write('\n')
'''

with open('./runs/'+run+'/'+run+'_reach_target_manualstart.txt','w') as target_outfile:
    target_outfile.write('#runstart_frame: '+str(runstart_frame)+'\n')
    for color, num in reach_target_prevedge.iteritems():
        target_outfile.write(color+'\t'+str(num)+'\t'+str(reach_target[color])+'\n')
    

time2=time()
runtime=time2-time1
print "Runtime: ", runtime
