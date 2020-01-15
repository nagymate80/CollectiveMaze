#!/usr/bin/python
import argparse

ap = argparse.ArgumentParser()

ap.add_argument("-or", "--p_orient", required=False,\
	help="Partner orientation when focal enters.")
ap.add_argument("-l", "--level", required=False,\
	help="Level of the crossings to examine.")
ap.add_argument("-n", "--node", required=False,\
	help="Chosen node.")
ap.add_argument("-g", "--gender", required=False,\
	help="Gender.")

args = vars(ap.parse_args())

if args["p_orient"] != None:
    p_orient = args["p_orient"]  
if args["level"] != None:
    given_level = int(args["level"])
if args["node"] != None:
    given_node = int(args["node"])
if args["gender"] != None:
    gender = args["gender"]

dnkrit = 0.3
               
level=[
    [0],
    [1,2,49],
    [3,4,50,51],
    [5,6,7,8],
    [9,10,11,12,13,14,15,16],
    [17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
    [33,34,35,36,37,38,38,40,41,42,43,44,45,46,47,48]
    ]

leveldic= {}
for levelindex, levellist in enumerate(level):
    for node in levellist:
        leveldic[node]=levelindex







res = {}
keys = ['in_alfa,water_b/g',\
        'in_alfa,water_alfa',\
        'in_b/g,water_alfa',\
        'in_b/g,water_back',\
        'in_b/g,wather_other_g/b']



partnerl = [['water_fw','other_fw','back'],\
         ['fw','back'],\
         ['water_fw','other_fw','back'],\
         ['fw_alfa','fw_notalfa','back'],\
         ['alfa','water_fw','back']]

focaloutl = [['water_fw','other_fw','back'],\
            ['to_partner_fw','not_partner_fw','back_to_water'],\
            ['water_fw','other_fw','back'],\
            ['fw_alfa','fw_notalfa','back'],\
            ['alfa','water_fw','back']]


for in0, i0 in enumerate(keys):
    res[i0] = {}
    for i1 in partnerl[in0]:
        res[i0][i1] = {}
        for i2 in focaloutl[in0]:
            res[i0][i1][i2] = 0

#print res

with open('dirnames_group','r') as runs:
    for run in runs:
        run = run[:-1]
        ###############################################
        if args["gender"] != None:
            if gender != run.split('_')[2][0]:
                continue
        ###############################################
        #print run
        with open('./runs/'+run+'/'+run+'_inteactions_stars_v2_'+str(dnkrit)+'.txt','r') as inter:
            for l in inter:
                if l[0]=='0':
                    print run
                if (l[0] == 'f') or (l[0]=='0'):
                    continue
                s = l.split('\t')
                node = int(s[1])

#############################################
                if args["node"] != None:
                    if node != given_node:
                        continue
                if args["level"] != None:
                    if leveldic[node] != given_level:
                        continue
                if args["p_orient"] != None:
                    if s[9] != p_orient:
                        continue
#############################################
         
                water = s[19]
                part = s[8]
                fin = s[12]
                fout = s[14]
                if fin == 'alfa':
                    if water != 'alfa':
                        if part == water:
                            if fout == water:
                                res['in_alfa,water_b/g']['water_fw']['water_fw'] += 1
                            elif fout != 'alfa':
                                res['in_alfa,water_b/g']['water_fw']['other_fw'] += 1
                            else:
                                res['in_alfa,water_b/g']['water_fw']['back'] += 1
                        elif part != 'alfa':
                            if fout == water:
                                res['in_alfa,water_b/g']['other_fw']['water_fw'] += 1
                            elif fout != 'alfa':
                                res['in_alfa,water_b/g']['other_fw']['other_fw'] += 1
                            else:
                                res['in_alfa,water_b/g']['other_fw']['back'] += 1
                        else: #part==back
                            if fout == water:
                                res['in_alfa,water_b/g']['back']['water_fw'] += 1
                            elif fout != 'alfa':
                                res['in_alfa,water_b/g']['back']['other_fw'] += 1
                            else:
                                res['in_alfa,water_b/g']['back']['back'] += 1

                    else: #water==alfa
                        if part != 'alfa':
                            if fout == part:
                                res['in_alfa,water_alfa']['fw']['to_partner_fw'] += 1
                            elif fout != 'alfa':
                                res['in_alfa,water_alfa']['fw']['not_partner_fw'] += 1
                            else:
                                res['in_alfa,water_alfa']['fw']['back_to_water'] += 1
                                
                        else: #part=='alfa'
                            if fout != 'alfa':
                                res['in_alfa,water_alfa']['back']['not_partner_fw'] += 1
                            else: #fout=='alfa'
                                res['in_alfa,water_alfa']['back']['back_to_water'] += 1
                            
                else: #fin!='alfa'
                    if water == 'alfa':
                        if part == water:
                            if fout == water:
                                res['in_b/g,water_alfa']['water_fw']['water_fw'] += 1
                            elif fout != fin:
                                res['in_b/g,water_alfa']['water_fw']['other_fw'] += 1
                            else:
                                res['in_b/g,water_alfa']['water_fw']['back'] += 1

                        elif part != fin:
                            if fout == water:
                                res['in_b/g,water_alfa']['other_fw']['water_fw'] += 1
                            elif fout != fin:
                                res['in_b/g,water_alfa']['other_fw']['other_fw'] += 1
                            else:
                                res['in_b/g,water_alfa']['other_fw']['back'] += 1

                        else: #part==fin
                            if fout == water:
                                res['in_b/g,water_alfa']['back']['water_fw'] += 1
                            elif fout != fin:
                                res['in_b/g,water_alfa']['back']['other_fw'] += 1
                            else:
                                res['in_b/g,water_alfa']['back']['back'] += 1

                    elif water == fin:
                        if part == 'alfa':
                            if fout == 'alfa':
                                res['in_b/g,water_back']['fw_alfa']['fw_alfa'] += 1
                            elif fout != fin:
                                res['in_b/g,water_back']['fw_alfa']['fw_notalfa'] += 1
                            else: #back
                                res['in_b/g,water_back']['fw_alfa']['back'] += 1

                        elif part != fin:
                            if fout == 'alfa':
                                res['in_b/g,water_back']['fw_notalfa']['fw_alfa'] += 1
                            elif fout != fin:
                                res['in_b/g,water_back']['fw_notalfa']['fw_notalfa'] += 1
                            else: #back
                                res['in_b/g,water_back']['fw_notalfa']['back'] += 1

                        else: #part==fin
                            if fout == 'alfa':
                                res['in_b/g,water_back']['back']['fw_alfa'] += 1
                            elif fout != fin:
                                res['in_b/g,water_back']['back']['fw_notalfa'] += 1
                            else: #back
                                res['in_b/g,water_back']['back']['back'] += 1

                    else: #water other g/b
                        if part == 'alfa':
                            if fout == 'alfa':
                                res['in_b/g,wather_other_g/b']['alfa']['alfa'] += 1
                            elif fout == water:
                                res['in_b/g,wather_other_g/b']['alfa']['water_fw'] += 1
                            else: #back
                                res['in_b/g,wather_other_g/b']['alfa']['back'] += 1

                        elif part == water:
                            if fout == 'alfa':
                                res['in_b/g,wather_other_g/b']['water_fw']['alfa'] += 1
                            elif fout == water:
                                res['in_b/g,wather_other_g/b']['water_fw']['water_fw'] += 1
                            else: #back
                                res['in_b/g,wather_other_g/b']['water_fw']['back'] += 1

                        else:
                            if fout == 'alfa':
                                res['in_b/g,wather_other_g/b']['back']['alfa'] += 1
                            elif fout == water:
                                res['in_b/g,wather_other_g/b']['back']['water_fw'] += 1
                            else: #back
                                res['in_b/g,wather_other_g/b']['back']['back'] += 1                            


#print res
gnlp= ''
for i in [args["gender"],args["node"],args["level"],args["p_orient"]]:
    if i != None:
        gnlp += ('_'+i)
    else:
        gnlp += '_None'

of = open('./hier/hier'+gnlp,'w')

for in0, i0 in enumerate(keys):
    #of.write('\n')
    of.write(i0)
    for i2 in focaloutl[in0]:
        of.write('\t'+i2)
    of.write('\n')
    for i1 in partnerl[in0]:
        of.write(i1)
        for i2 in focaloutl[in0]:
            of.write('\t'+str(res[i0][i1][i2]))
        of.write('\n')
    of.write('\n')




of.close()

                                

