

dnkrit = 0.3
mfix_notallowed = 0


n_o_t = {9:{'beta': 1, 'gamma': 2}, 10:{'beta': 3, 'gamma': 4}, 11:{'beta': 5, 'gamma': 6}, \
         12:{'beta': 7, 'gamma': 8}, 13:{'beta': 9, 'gamma': 10}, 14:{'beta': 11, 'gamma': 12}, \
         15:{'beta': 13, 'gamma': 14}, 16:{'beta': 15, 'gamma': 16} }




with open('/home/hattila/Patek/atlasz/ratmaze/video/edgecount/dirnames/dirnames_group','r') as runs:
    for run in runs:
        run = run[:-1]
        target = int(run.split('target')[1])
        #rround = int(run.split('_')[0][6:])
        #if rround < 8:
        #   continue
        patek_endpts = {}
        
        with open('/home/hattila/Patek/atlasz/ratmaze/video/edgecount/runs/'+run+'/'+run+'_individuals_stars_v6_dnkrit_'+str(dnkrit)+'_mfix_'+str(mfix_notallowed)+'.txt','r') as inter:
            for l in inter:
                if l[0] == 'f':
                    continue
                s = l.split('\t')
                n = int(s[1])
                color = s[2]
                inn = s[3]
                out = s[5]
                
                if inn != 'alfa':
                    continue
                if out == 'alfa':
                    continue
                if n < 9:
                    continue

                if color not in patek_endpts:
                    patek_endpts[color] = []

                t = str(n_o_t[n][out])
                patek_endpts[color].append(t)
                

        for color in patek_endpts:
            print run, color, target, ' '.join(patek_endpts[color])
                
