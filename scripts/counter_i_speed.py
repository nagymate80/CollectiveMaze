import numpy

dnkrit = 0.3
mfix_notallowed = 784

def avg_and_std(values): 
    average = numpy.average(values)
    std = numpy.std(values)  
    return (average, std)


M12=['ORBM','OGBM','OBGM','GRBM','GRPM','GPBM','BRPM','BGPM']
M34=['ORPM','OBPM','OPGM','OPBM','GOBM','GOPM','GBPM','BOPM']
F12=['ROGF','ROBF','RGOF','RGBF','RGPF','ORBF','OGPF','GRBF']
F34=['ROPF','RBOF','RBGF','RBPF','RPOF','RPGF','RPBF','ORGF']

dist = {}
frames = {}
speed_group = {}
speed_single= {}

for group in [M12,M34,F12,F34]:
    for color in group:
        dist[color]=0.0
        frames[color]=0
        speed_group[color] = []
        speed_single[color] = []

s = []
s_prev = []

dirnames = ['dirnames_single80', 'dirnames_group']

for dirname in dirnames:
    with open('/home/attila/PATEKL/video/edgecount/dirnames/'+dirname,'r') as runs:
        for rind, run in enumerate(runs):
            run = run[:-1]

            with open('/home/attila/PATEKL/video/edgecount/runs/'+run+'/'+run+'_proj','r') as proj:
                pateknames = []
                for lindex, line in enumerate(proj):
                    if line[0] == 'f':
                        continue
                    if lindex < 15:
                        continue
                    elif lindex == 15:
                        s = line.split('\t')
                        pateknum = int(s[1])
                        for p in range(pateknum):
                            color = s[2+p*15]
                            pateknames.append(color)
                            dist[color]=0.0
                            frames[color]=0
                    else:
                        s_prev = s
                        s = line.split('\t')

                        
                        for p in range(pateknum):
                            mfix = int(s[16+p*15])
                            mfix_prev = int(s_prev[16+p*15])
                            if ((mfix & mfix_notallowed) == 0) and ((mfix_prev & mfix_notallowed) == 0):
                                edge = s[7+p*15]
                                edge_prev = s_prev[7+p*15]
                                dnorm = float(s[9+p*15])
                                dnorm_prev = float(s_prev[9+p*15])
                                if (edge == edge_prev) and (1 > dnorm_prev) and (1 > dnorm):
                                    dd = abs(dnorm-dnorm_prev)
                                    if dd > 0.01:
                                        color = s[2+p*15]
                                        dist[color] += dd
                                        frames[color] += 1
                    
                for color in pateknames:
                    v = dist[color] / frames[color]
                    if v > 1.0:
                        print run, color, v
                    if (dirname == 'dirnames_single80'):
                        (speed_single[color]).append(v)
                    elif (dirname == 'dirnames_group'):
                        (speed_group[color]).append(v)
            
'''
for key, value in speed:
    print key,
    for i in value:
        print i,
    print ''

for group in [M12,M34,F12,F34]:
    for color in group:
        print color,
        for v in speed[color]:
            print v,
        print ''
'''

for group in [M12,M34,F12,F34]:
    for color in group:
        print color, numpy.average(speed_single[color]), numpy.std(speed_single[color]), \
              numpy.average(speed_group[color]), numpy.std(speed_group[color])












