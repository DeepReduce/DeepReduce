import os
from collections import defaultdict
import numpy as np

def tree(): return defaultdict(tree)

f = open('modify_summary.tsv')
content = f.read()
f.close()

content = content.splitlines()
result = {}

f = open('modify_summary.csv','w')
for item in content:
    templist = item.split('\t')
    templist[0] = '_'.join(templist[0].lstrip('nin_').split('_')[0:-1])
    if templist[0] in result:
        result[templist[0]].append(eval(templist[1]))
    else:
        result[templist[0]] = [eval(templist[1])]
    #print(templist)
    f.write(','.join(templist) + '\n')
f.close()

for item in result:
    print(result[item])
    print('%s : %s'%(item,np.mean(result[item])))
