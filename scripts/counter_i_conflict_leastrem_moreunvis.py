#!/usr/bin/python

dnkrit = 0.3
mfix_notallowed = 0

least_rem = 0
more_unvis = 0

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

                beta_notvis = int(s[19])
                gamma_notvis = int(s[21])

                if beta_notvis == gamma_notvis:
                    continue

                if beta_notvis > gamma_notvis:
                    m_unvis = 'beta'
                else:
                    m_unvis = 'gamma'

                if l_rem == m_unvis:
                    continue

                if out == l_rem:
                    least_rem += 1

                else:
                    more_unvis += 1

for i in ['least_rem', 'more_unvis']:
    print i, eval(i)

            
               





            
