import os
import math
import copy
from tqdm import tqdm
import sys

#global K
K = 20


def readFile(filepath):
    f = open(filepath)
    content = f.read()
    f.close()
    return content.splitlines()

def getDistribute(outputs,NSec):
    nnum = len(outputs[0])
    tnum = len(outputs)
    NDist = []
    for i in range(nnum):
        temps = [0] * K
        for j in range(tnum):
            tempv = outputs[j][i]
            tempmin = NSec[i][0]
            tempmax = NSec[i][1]
            tempindex = getK(tempmin,tempmax,tempv)
            try:
                temps[tempindex-1] += 1
            except:
                print(tempindex)
                input('getDIstribute error ...')
        for k in range(K):
            temps[k] = temps[k]/tnum
        NDist.append(copy.deepcopy(temps))
        #print(sum(temps))
        #input('check...')
    return NDist


def getSectionIndex(outputs,NSec):
    nnum = len(outputs[0])
    tnum = len(outputs)
    NIndex = []
    for j in range(tnum):
        temps = [0] * nnum
        for i in range(nnum):
            tempv = outputs[j][i]
            tempmin = NSec[i][0]
            tempmax = NSec[i][1]
            tempindex = getK(tempmin,tempmax,tempv)
            temps[i] = tempindex -1
        NIndex.append(copy.deepcopy(temps))
    return NIndex


def getSection(outputs):
    NSec = []
    #print(outputs[0])
    #print(len(outputs[0]))
    nnum = len(outputs[0])
    tnum = len(outputs)
    for i in range(nnum):
        omax = outputs[0][i]
        omin = outputs[0][i]
        for j in range(tnum):
            try:
                omax = max(omax,outputs[j][i])
                omin = min(omin,outputs[j][i])
            except:
                print(nnum)
                print(len(outputs[j]))
                print("%s : %s"%(j,i))
                print(outputs[j])
                input('getSection error check...')
        NSec.append((omin,omax))
    return NSec

def getK(tmi,tma,tm):
    step = (tma-tmi)/K
    index = int(math.ceil((tm-tmi)/step))
    if index == 0:
        return index+1
    else:
        return index




if __name__ == '__main__':
    #changelayer = int(sys.argv[1])
    #prepath = 'doubleneuronnumber-lenet-1/' + str(changelayer) + '/'

    #path = prepath + 'Cov/'
    #path1 = path
    threshold = '0.5'
    path1 = os.getcwd() + '/Cov/'
    path = os.getcwd() + '/Cov/activeneuron/' + threshold + 'ase/'
    noutput = readFile(path1 + 'cross_entropy')
    for i in tqdm(range(len(noutput))):
        noutput[i] = eval(noutput[i])
    NSec = getSection(noutput)
    #NDist = getDistribute(noutput,NSec)
    NIndex = getSectionIndex(noutput,NSec)
    f = open(path + 'kcoverage','w')
    for i in tqdm(range(len(NIndex))):
        for j in range(len(NIndex[i])):
            f.write(str(NIndex[i][j]) + ',')
        f.write('\n')
    f.close()
