#!/usr/bin/python

dnkrit = 0.3
mfix_notallowed = 0

more_unvis = 0
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

               

                inn = s[3]
                out = s[5]

                if inn != 'alfa':
                    continue

                if inn == out:
                    continue


                beta_notvis = int(s[19])
                gamma_notvis = int(s[21])

                if beta_notvis == gamma_notvis:
                    continue

                if beta_notvis > gamma_notvis:
                    m_unvis = 'beta'
                else:
                    m_unvis = 'gamma'

               

                beta_num = int(s[14])
                gamma_num = int(s[15])

                if beta_num == gamma_num:
                    continue

                if beta_num > gamma_num:
                    m_partners = 'beta'
                else:
                    m_partners = 'gamma'

                if m_unvis == m_partners:
                    continue

                if out == m_unvis:
                    more_unvis += 1

                else:
                    more_partners += 1

for i in ['more_unvis', 'more_partners']:
    print i, eval(i)

            
               





            
