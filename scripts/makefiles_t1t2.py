#!/usr/bin/python
import sys
from os import system

levels = ['2','3','4','None']
nodes = []
for number in range(3,17):
    nodes.append(str(number))
nodes.append('None')
genders = ['M','F','None']

for l in levels:
    if l != 'None':
        for g in genders:
            par = "python counter_i_tbigger_tsmaller.py -l "+l
            if g != 'None':
                par += " -g "+g
            system(par)
    else:
        for n in nodes:
            for g in genders:
                par = "python counter_i_tbigger_tsmaller.py"
                if n != 'None':
                    par += " -n "+n
                if g != 'None':
                    par += " -g "+g
                system(par)            
