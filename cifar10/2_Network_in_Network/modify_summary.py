import os
import sys

fs = os.listdir('input/modify/predict/')

keyword = sys.argv[1]

g = open('result/DeepReduce/0.5/DRcov-20-last-layer.result')
line = g.readline()
origin = eval(line)
g.close()

ours = []
oris = []
deltas = []

for f in fs:
    if keyword in f:
        with open('input/modify/predict/' + f) as g:
            our = 0
            ori = 0
            lines = g.readlines()
            for line in lines:
                if int(eval(line)[0]) == 1:
                    ori += 1
            for ii in origin:
                if int(eval(lines[ii])[0]) == 1:
                    our += 1
        ori = ori/float(10000)
        our = our/float(len(origin))
        delta = abs(ori - our)
        ours.append(our)
        oris.append(ori)
        deltas.append(delta)
        print ("%s\t%f\t%f\t%f" % (f, ori, our, delta))

#print (len(oris))

#print (sum(oris)/len(oris))

#print (sum(ours)/len(ours))

#print (sum(deltas)/len(deltas))


