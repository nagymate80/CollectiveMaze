dirnames=open('dirnames_group', 'r')
osszpatek=8
'''
M12=['ORBM','OGBM','OBGM','GRBM','GRPM','GPBM','BRPM','BGPM']
M34=['ORPM','OBPM','OPGM','OPBM','GOBM','GOPM','GBPM','BOPM']
F12=['ROGF','ROBF','RGOF','RGBF','RGPF','ORBF','OGPF','GRBF']
F34=['ROPF','RBOF','RBGF','RBPF','RPOF','RPGF','RPBF','ORGF']

M12_lg=['ORBM','OGBM','OBGM','GRPM','BRPM','BGPM']
M34_lg=['ORPM','OPGM','GOBM','GOPM','GBPM','BOPM']
F12_lg=['ROGF','ROBF','RGOF','RGBF','RGPF','ORBF']
F34_lg=['RBOF','RBGF','RBPF','RPOF','RPGF','ORGF']
'''

for target in range(1,17):
    outfile = open('./targets/'+str(target)+'.txt','w')
    outfile.write('round\t')
    for i in range(1,1+osszpatek):
        outfile.write(str(i)+'\t')
    outfile.write('average')
    outfile.close()


for line1 in dirnames:
    run=line1[:-1]
    runsplit=run.split('_')
    target = runsplit[3][6:]
    roundd = runsplit[0][5:]
    reachfile = open('./runs/'+run+'/'+run+'_reach_target_manualstart.txt','r')
    outfile = open('./targets/'+target+'.txt','a')

    reach_frames = []
    
    for line2 in reachfile:
        if line2[0] == '#':
            continue
        reach_frames.append(int(line2.split('\t')[2]))
    reach_frames.sort()
    sum_reach_frames = sum(reach_frames)
    outfile.write('\n')
    outfile.write(roundd+'\t')
    for i in reach_frames:
        outfile.write(str(i)+'\t')
    outfile.write(str(int(sum(reach_frames)/len(reach_frames))))
    
                    


    reachfile.close()
    outfile.close()

dirnames.close()
