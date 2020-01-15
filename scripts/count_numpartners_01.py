

dnkrit = 0.3
mfix_notallowed = 0

to_more = 0
to_less = 0
eq = 0
no_partners = 0

res = {x:{0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0} for x in range(8)}

abg_id = {'alfa':0, 'beta':1, 'gamma':2}

def rlb(back, dir2): #right-left-back
    if back == 'alfa':
        if dir2 == 'alfa':
            return 'back'
        elif dir2 == 'beta':
            return 'right'
        elif dir2 == 'gamma':
            return 'left'

    elif back == 'beta':
        if dir2 == 'alfa':
            return 'left'
        elif dir2 == 'beta':
            return 'back'
        elif dir2 == 'gamma':
            return 'right'

    elif back == 'gamma':
        if dir2 == 'alfa':
            return 'right'
        elif dir2 == 'beta':
            return 'left'
        elif dir2 == 'gamma':
            return 'back'






with open('dirnames_group','r') as runs:
    for run in runs:
        #if run.split('_')[2][0] != e0:
        #    continue
        run = run[:-1]
        #print run
        with open('./runs/'+run+'/'+run+'_individuals_stars_v2_dnkrit_'+str(dnkrit)+'_mfix_'+str(mfix_notallowed)+'.txt','r') as inter:        
            for l in inter:
                if (l[0] == 'f') or (l[0] == '0'):
                    continue
                s = l.split('\t')

                inn = s[3]
                in_id = abg_id[inn]
                out = s[5]

                if out != inn:
                    abg = ['alfa','beta','gamma']
                    abg.remove(inn)
                    nums = [int(s[13]), int(s[14]),int(s[15])]
                    del nums[in_id]

                    
                    if out == abg[0]:
                        out_id2 = 0
                        other_id2 = 1
                    else:
                        out_id2 = 1
                        other_id2 = 0
                        

                    if (nums[0] == 0) and (nums[1] == 0):
                        no_partners += 1
                        
                    elif nums[0] == nums[1]:
                        eq += 1
                        
                    elif nums[0] > nums[1]:
                        if out == abg[0]:
                            to_more += 1
                            
                        else:
                            to_less += 1
                            
                    else:
                        if out == abg[0]:
                            to_less += 1
                        else:
                            to_more += 1


                    res[nums[out_id2]][nums[other_id2]] += 1

print 'to_more', to_more
print 'to_less', to_less
print 'eq', eq
print 'no_partners_fw', no_partners
print '\n'
print 'out\tother\tfound\t\t\tout\tother\tfound'
                    
for nout in range(8):
    for nother in range(8):
        print nout, '\t', nother, '\t', res[nout][nother], '\t',
        if (nout==0) and (nother==0):
            print '\t',
        else:
            if (res[nout][nother]!=0) or (res[nother][nout]!=0):
                elso = float(res[nout][nother])/float((res[nout][nother]+res[nother][nout]))
                print elso, '\t', 1-elso,
            else:
                print '\t',
            


        print '\t', nother, '\t', nout, '\t', res[nother][nout]




