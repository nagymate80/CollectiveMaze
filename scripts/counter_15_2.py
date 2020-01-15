'''two pateks coming opposite'''

dnkrit = 0.3
mfix_notallowed = 0

events = ['together_third', 'together_one\'s_entry', 'both_to_other\'s_entry',\
          'one_third_one_back', 'p1_third_p2_other\'sentry','p2_third_p1_other\'sentry', 'both_back']
ered = {}
for i in events:
    ered[i] = 0
notvalid = 0

level=[
    [0],
    [1,2,49],
    [3,4,50,51],
    [5,6,7,8],
    [9,10,11,12,13,14,15,16],
    [17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
    [33,34,35,36,37,38,38,40,41,42,43,44,45,46,47,48]
    ]

leveldic= {}
for levelindex, levellist in enumerate(level):
    for node in levellist:
        leveldic[str(node)]=levelindex

with open('dirnames_group','r') as runs:
    for run in runs:
        run = run[:-1]
        with open('./runs/'+run+'/'+run+'_inteactions_stars_opp_dnkrit_'+str(dnkrit)+'_mfix_'+str(mfix_notallowed)+'.txt','r') as inter:
            for l in inter:
                if l[0] == 'f':
                    continue
                s = l.split('\t')

                node = s[1]
                p1in=s[4]
                p1in_or = s[5]
                p1out = s[6]
                p1out_or = s[7]
                p2in=s[10]
                p2in_or = s[11]
                p2out = s[12]
                p2out_or = s[13]                

                if (p1in_or == '-1') and (p2in_or == '-1') and (p1out_or == '1') and (p2out_or == '1') and (p1in != p2in):
                    if (p1out != p1in) and (p1out != p2in) and  (p2out != p2in) and (p2out != p1in):
                        ered['together_third'] += 1
                    elif ((p1out == p1in) and (p2out == p1in)) or ((p1out == p2in) and (p2out == p2in)):
                        ered['together_one\'s_entry'] += 1
                    elif ((p1out == p2in) and (p2out == p1in)):
                        ered['both_to_other\'s_entry'] += 1
                    elif ((p1out != p1in) and (p1out != p2in) and (p2out == p2in)) or ((p2out != p2in) and (p2out != p1in) and (p1out == p1in)):
                        ered['one_third_one_back'] += 1
                    elif ((p1out != p1in) and (p1out != p2in) and (p2out == p1in)):
                        ered['p1_third_p2_other\'sentry'] += 1
                    elif ((p2out != p2in) and (p2out != p1in) and (p1out == p2in)):
                        ered['p2_third_p1_other\'sentry'] += 1
                    elif (p1out == p1in) and (p2out == p2in):
                        ered['both_back'] += 1

                else:
                    notvalid += 1

for i in events:
    print i, ered[i]

print 'sum events:', sum(ered.values())
print 'notvalid:', notvalid















                    

