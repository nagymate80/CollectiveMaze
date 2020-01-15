#!/usr/bin/python

dnkrit = 0.3
mfix_notallowed = 0

back_most_notvis = 0
nott = 0



with open('dirnames_group','r') as runs:
    for run in runs:
        run = run[:-1]
        #print run
        with open('./runs/'+run+'/'+run+'_individuals_stars_v3_dnkrit_'+str(dnkrit)+'_mfix_'+str(mfix_notallowed)+'.txt','r') as inter:
            for l in inter:
                if l[0] == 'f':
                    continue
                s = l.split('\t')
                inn = s[3]
                out = s[5]
                if out == inn:
                    abg_id = {'alfa':0,'beta':1,'gamma':2}
                    out_notvis = int(s[17+2*abg_id[out]]) 
                    
                    
                    del abg_id[out]
                    other0_notvis = int(s[17+2*abg_id.values()[0]]) 
                    other1_notvis = int(s[17+2*abg_id.values()[1]]) 
                    
                    if (out_notvis > other0_notvis) and (out_notvis > other1_notvis):
                        back_most_notvis += 1
                    else:
                        nott += 1
                    
for i in ['back_most_notvis','nott']:
    print i, eval(i)
