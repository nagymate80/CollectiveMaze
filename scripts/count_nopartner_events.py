

dnkrit = 0.3
mfix_notallowed=0

ered = {}

list1 = ['alfa','beta','gamma']
list2 = ['alfa','beta','gamma']
list3 = ['alfa','beta','gamma']

for elem1 in list1:
    for elem2 in list2:
        for elem3 in list3:
            ered[elem1+'_'+elem2+'_'+elem3] = 0

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
'''
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
'''
osszes=0
with open('./dirnames/dirnames_group','r') as runs:
    for run in runs:
        run = run[:-1]
        #print run
        with open('./runs/'+run+'/'+run+'_individuals_stars_v5_dnkrit_'+str(dnkrit)+'_mfix_'+str(mfix_notallowed)+'.txt','r') as inter:
            for l in inter:
                if l[0] == 'f':
                    continue
                s = l.split('\t')
                if float(s[28]) > 2.3:
                    osszes+=1
print osszes

