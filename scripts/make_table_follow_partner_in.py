p_or = 'None'

of = open('interact_sum_or'+p_or+'.txt','w')
of.write('Partner_orientation:'+'\t'+p_or+'\n\n')

levels = ['2','3','4']

for l in levels:
    of.write('Level '+l+'\t\t\t\t\t')
of.write('\n')
    
for gender in ['None','M','F']:
    for l in levels:
        of.write('In, water'+'\t\t\t\t\t')
    of.write('\n')

    infil = [open('./hier/hier_'+gender+'_None_'+l+'_'+str(p_or),'r') for l in levels]


    while(True):
        for fil in infil:
            line = fil.readline()
            if line == '':
                break
            else:
                of.write(line[:-1]+'\t\t')
        of.write('\n')
        if line == '':
            break
                
    for fil in infil:
        fil.close()

    of.write('\n\n')

of.close()
