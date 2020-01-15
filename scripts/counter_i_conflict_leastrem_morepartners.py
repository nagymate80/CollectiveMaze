#!/usr/bin/python

dnkrit = 0.3
mfix_notallowed = 0

least_rem = 0
more_partners = 0

with open('/home/attila/PATEKL/video/edgecount/dirnames/dirnames_group','r') as runs:
    for run in runs:
        run = run[:-1]
        with open('/home/attila/PATEKL/video/edgecount/runs/'+run+'/'+run+'_individuals_stars_v6_dnkrit_'+str(dnkrit)+'_mfix_'+str(mfix_notallowed)+'.txt','r') as inter:
            for l in inter:
                if l[0] == 'f':
                    continue
                s = l.split('\t')

                isdrunk = s[8]
                if isdrunk == 'False':
                    continue
                

                alfa_ago = int(s[22])
                beta_ago = int(s[23])
                gamma_ago = int(s[24])

                minus_one = 0
                abg_l = ['alfa', 'beta', 'gamma']
                for a in abg_l:
                    if eval(a+'_ago') == -1:
                        minus_one += 1

                if minus_one == 2:
                    continue

                inn = s[3]
                out = s[5]

                if inn != 'alfa':
                    continue

                if inn == out:
                    continue

                if (beta_ago == -1) or (beta_ago > gamma_ago):
                    l_rem = 'beta'
                else:
                    l_rem = 'gamma'

                beta_num = int(s[14])
                gamma_num = int(s[15])

                if beta_num == gamma_num:
                    continue

                if beta_num > gamma_num:
                    m_partners = 'beta'
                else:
                    m_partners = 'gamma'

                if l_rem == m_partners:
                    continue

                if out == l_rem:
                    least_rem += 1

                else:
                    more_partners += 1

for i in ['least_rem', 'more_partners']:
    print i, eval(i)

            
               





            
