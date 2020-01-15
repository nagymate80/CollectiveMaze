

dnkrit = 0.3
mfix_notallowed = 0

for node in range(3,17):
    for given in ['beta','gamma']:

        other_l = ['beta','gamma'] 

        other_l.remove(given)
        other= other_l[0]


        abg_l = ['alfa','beta','gamma']
        abg_id = {'alfa':0,'beta':1,'gamma':2}

        res = {'partner':{'water':{'out_beta':0,'out_gamma':0},'nowater':{'out_beta':0,'out_gamma':0}}, \
               'nopartner':{'water':{'out_beta':0,'out_gamma':0},'nowater':{'out_beta':0,'out_gamma':0}}}



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

        def ispartner(direction, splitted_line):
            if int(splitted_line[ 13+abg_id[direction] ]) == 0:
                return 'nopartner'
            else:
                return 'partner'

        def iswater(direction, splitted_line):
            if splitted_line[9] == direction:
                return 'water'
            else:
                return 'nowater'

        with open('/home/attila/PATEKL/video/edgecount/dirnames/dirnames_group','r') as runs:
            for run in runs:
                run = run[:-1]
               
                with open('/home/attila/PATEKL/video/edgecount/runs/'+run+'/'+run+'_individuals_stars_v6_dnkrit_'+str(dnkrit)+'_mfix_'+str(mfix_notallowed)+'.txt','r') as inter:
                    for l in inter:
                        if l[0] == 'f':
                            continue
                        s = l.split('\t')

                        n= int(s[1])
                        if n != node:
                            continue

                        inn = s[3]
                        if inn != 'alfa':
                            continue
                        out = s[5]
                        if out == 'alfa':
                            continue

                        res[ispartner(given, s)][iswater(given, s)]['out_'+out] += 1

        #print res



        print 'Node', node, 'given', given, '\n'
        print '\tpartner_on_'+given, 'no_partner_on_'+given
        print 'water_on_'+given,
        print "{0:.0f}%".format(100*float(res['partner']['water']['out_'+given]) / (float(res['partner']['water']['out_'+other]) + float(res['partner']['water']['out_'+given]))),
        print "{0:.0f}%".format(100*float(res['nopartner']['water']['out_'+given]) / (float(res['nopartner']['water']['out_'+other]) + float(res['nopartner']['water']['out_'+given])))
        print 'no_water_on_'+given,    
        print "{0:.0f}%".format(100*float(res['partner']['nowater']['out_'+given]) / (float(res['partner']['nowater']['out_'+other]) + float(res['partner']['nowater']['out_'+given]))),
        print "{0:.0f}%".format(100*float(res['nopartner']['nowater']['out_'+given]) / (float(res['nopartner']['nowater']['out_'+other]) + float(res['nopartner']['nowater']['out_'+given])))
        print '\n\n'


               


