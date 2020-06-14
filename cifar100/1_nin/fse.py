import os
import copy
import random
import math
import numpy as np
#import keras
#from keras import optimizers
#from keras.datasets import cifar10
#from keras.models import Sequential
#from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
#from keras.callbacks import LearningRateScheduler, TensorBoard
#from keras.models import load_model
from collections import defaultdict
import numpy as np
import sys
from tqdm import tqdm

def tree(): return defaultdict(tree) 
sys.setrecursionlimit(1000000)

def readFile(filepath):
    f = open(filepath)
    content = f.read()
    f.close()
    return content.splitlines()


predict = readFile('input/predict')
for i in range(len(predict)):
    predict[i] = eval(predict[i])

#print(predict)

def calculate_acc(model1,RST):
    count = 0
    acc = 0
    predict = []
    for i in range(len(RST)):
        #test_image = x_test[RST[i]].reshape([1,32,32,3])
        #y = model1.predict_classes(test_image)
        y = model1[RST[i]][0]
        if y == 1:
            acc += 1
            #predict.append((1,y[0],np.argmax(y_test[RST[i]])))
        else:
            #predict.append((0,y[0],np.argmax(y_test[RST[i]])))
            pass
        count += 1
    return acc/count


global KN
KN = 20

def readFile(filepath):
    f = open(filepath)
    content = f.read()
    f.close()
    return content.splitlines()




def update_dist(RST,RS,RSTDist,outputs,NIndex,NMap):
    for i in RS:
        output = NIndex[i]
        for j in range(len(RSTDist)):
            tempindex = output[j]
            RSTDist[j][tempindex] += 1
    return RSTDist


# ds : distribution of sample set; dt : distribution of test set
def calculate_ce(ds,dt,snum):
    eps = 1e-15
    if snum == 0:
        #eps = 1e-15
        cesum = 0
        #print(len(ds))
        for i in range(len(ds)):
            for k in range(KN):
                cesum -= dt[i][k] * np.log(eps)
    else:
        cesum = 0
        for i in range(len(ds)):
            for k in range(KN):
                if ds[i][k] != 0:
                    cesum -= dt[i][k] * np.log(ds[i][k]/snum)
                else:
                    cesum -= dt[i][k] * np.log(eps)
    return cesum/len(ds)

# ds : distribution of sample set; dt : distribution of test set
def calculate_kl(ds,dt,snum):
    eps = 1e-15
    if snum == 0:
        #eps = 1e-15
        cesum = 0
        #print(len(ds))
        for i in range(len(ds)):
            for k in range(KN):
                cesum += dt[i][k] * np.log(dt[i][k]/eps)
    else:
        cesum = 0
        for i in range(len(ds)):
            for k in range(KN):
                #if ds[i][k] != 0:
                cesum += max(dt[i][k],eps) * np.log(max(dt[i][k],eps)/max(ds[i][k]/snum,eps))
                #else:
                #    cesum -= dt[i][k] * np.log(eps)
    return cesum/len(ds)

def cce(ds,dt):
    eps = 1e-15
    cesum = 0
    for i in range(len(ds)):
        for k in range(KN):
            #if ds[i][k]!= 0:
            cesum += max(dt[i][k],eps) * np.log(max(dt[i][k],eps)/max(ds[i][k],eps))
            #else:
            #    cesum += dt[i][k] * np.log(eps/eps)
    return cesum/len(ds)


def fse(fseter,NDist,outputs,NIndex,NMap):
    RST = []
    log = []
    #default_cov = getCoverage(mapdict,[])
    dnum = len(NDist)
    #print(default_cov)
    # empty distribute set
    RSTDist = []
    #print('nnum : %s'%dnum)
    for i in range(dnum):
        RSTDist.append([0] * KN)
    default_ce = cce(NDist,NDist)
    #Distribute = getDistrubute()
    #print('the default cross entropy : %s'%default_ce)
    default_acc = calculate_acc(predict,range(10000))
    #print('the default accuracy : %s'%default_acc)
    #input('check...')
    #terminal_cov = 0
    terminal_ce =  float('inf')
    RS = random.sample(range(10000),30)
    RST.extend(copy.deepcopy(RS))
    RSTDist = update_dist(RST,RS,RSTDist,outputs,NIndex,NMap)
    while(len(RST) < fseter):
        q = min(5,fseter-len(RST))
        examples = []
        remain = list(set(range(10000)) - set(RST))
        for i in range(300): 
            examples.append(random.sample(remain,q))
        mince = 100000
        minindex = -1
        for i in range(300):
            example = examples[i]
            exdist =  update_dist(RST,example,copy.deepcopy(RSTDist),outputs,NIndex,NMap)
            tempce = calculate_ce(exdist,NDist,len(RST)+len(example))
            #print(RSTDist)
            if tempce < mince:
                mince = tempce
                minindex = i
        RS = copy.deepcopy(examples[minindex])
        RST.extend(copy.deepcopy(RS))
        RSTDist = update_dist(RST,RS,RSTDist,outputs,NIndex,NMap)
        tempkl = calculate_kl(RSTDist,NDist,len(RST))
        #print(minkl)
        #print(minindex)
    #print(calculate_kl(RSTDist,NDist,len(RST)))
    #print(calculate_acc(predict,RST))
    #print(getCoverage(mapdict,RST))
    #print('**************')
    #input('check...')
    return RST,tempkl,calculate_acc(predict,RST)


def kllog(RSTDist,NDist,RST):
    ft = open('klt','a')
    fr = open('klr','a')
    frst = open('klrst','a')
    rnum = len(RST)
    rtt = copy.copy(RSTDist)
    ft.write(str(NDist) + '\n')
    for i in range(len(RSTDist)):
        for j in range(len(RSTDist[i])):
            rtt[i][j] = rtt[i][j]/rnum
    fr.write(str(rtt) + '\n')
    frst.write(str(RST) + '\n')
    frst.close()
    ft.close()
    fr.close()

def checksum(RSTDist,rnum):
    for i in range(len(RSTDist)):
        if sum(RSTDist[i]) == rnum:
            continue
        else:  
            print("length of RSTDist : %s, length of RST : %s"%(sum(RSTDist[i]),rnum)) 
            print('checksum error ...')
            input('pause...') 


def getCandidate(RSTDist,NDist,NMap,mapdict,tnum,RST):
    maxindex = (0,0)
    maxdiff = NDist[0][0]*tnum - RSTDist[0][0]
    for i in range(len(RSTDist)):
        for j in range(len(RSTDist[i])):
            tempdiff =  NDist[i][j]*tnum - RSTDist[i][j]
            if tempdiff > maxdiff:
                maxdiff = tempdiff
                maxindex = (i,j)
    templist = copy.deepcopy(NMap[maxindex[0]][maxindex[1]])
    templist = list(set(templist) - set(RST))
    #print('candidate Neuron Section : %s'%str(maxindex))
    return random.sample(templist,1)

def getCandidate_new(RSTDist,NDist,NMap,mapdict,tnum,RST,NIndex):
    relist = []
    for i in range(len(RSTDist)):
        maxindex = (i,0)
        maxdiff = NDist[i][0]*tnum - RSTDist[i][0]
        for j in range(len(RSTDist[i])):
            tempdiff = NDist[i][j]*tnum - RSTDist[i][j]
            if tempdiff > maxdiff:
                maxdiff = tempdiff
                maxindex = (i,j)
        relist.append(maxindex[1])
    maxsim = 0
    remain = list(set(range(10000))- set(RST))
    simlist = [0] * len(remain)
    simdict = {}
    for i in range(len(remain)):
        simlist[i] = calSimilarity(NIndex[remain[i]],relist)
        maxsim = max(simlist[i],maxsim)
        if simlist[i] in simdict.keys():
            simdict[simlist[i]].append(remain[i])
        else:
            simdict[simlist[i]] = [remain[i]]
    return random.sample(simdict[maxsim],1)
        

def calSimilarity(a,b):
    resim = 0
    for i in range(len(a)):
        if a[i] == b[i]:
            resim += 1
    return resim 

def getCandidate_negative(RSTDist,NDist,NMap,mapdict,tnum,RST,NIndex):
    relist = []
    for i in range(len(RSTDist)):
        maxindex = (i,0)
        maxdiff = RSTDist[i][0] - NDist[i][0]*tnum
        for j in range(len(RSTDist[i])):
            tempdiff = RSTDist[i][0] - NDist[i][0]*tnum
            if tempdiff > maxdiff:
                maxdiff = tempdiff
                maxindex = (i,j)
        relist.append(maxindex[-1])
    maxsim = 0
    remain = list(set(range(10000))- set(RST))
    simlist = [0] * len(remain)
    simdict = {}
    for i in range(len(remain)):
        simlist[i] = calSimilarity(NIndex[remain[i]],relist)
        maxsim = max(simlist[i],maxsim)
        if simlist[i] in simdict.keys():
            simdict[simlist[i]].append(remain[i])
        else:
            simdict[simlist[i]] = [remain[i]]
    return random.sample(simdict[maxsim],1)


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
    step = (tma-tmi)/KN
    index = math.ceil((tm-tmi)/step)
    if index == 0:
        return index+1
    else:
        return index
    

def getDistribute(outputs,NSec):
    nnum = len(outputs[0])
    tnum = len(outputs)
    NDist = []
    for i in range(nnum):
        temps = [0] * KN
        for j in range(tnum):
            tempv = outputs[j][i]
            tempmin = NSec[i][0]
            tempmax = NSec[i][1]
            tempindex = getK(tempmin,tempmax,tempv)
            temps[tempindex-1] += 1
        for k in range(KN):
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

def getNeuronMap(NIndex):
    relist = tree()
    for i in range(len(NIndex[0])):
        for j in range(KN):
            relist[i][j] = set()
    for i in range(len(NIndex)):
        for j in range(len(NIndex[i])):
            tempindex = NIndex[i][j]
            relist[j][tempindex].add(i)
    return relist 

testdict = {0:{0,4},
            1:{4},
            2:{0,1,2},
            3:{2,5},
            4:{0,3},
            5:{0,5},
            6:{2,3,6},
            7:{1,2,3,6}}


if __name__ == "__main__":
    #path = os.getcwd() 
    #threshold = str(sys.argv[1])
    fseter = int(sys.argv[1])
    runN = int(sys.argv[2])
    threshold = float(sys.argv[3])
    path1 = os.getcwd() + '/input/'
    path2 = os.getcwd() + '/result/FSE_new/' + str(threshold) + '/'
    if os.path.exists(path2) == False:
        os.makedirs(path2)
    #path = os.getcwd() + '/Cov/activeneuron/' + threshold + 'ase/'
    #cov = readFile(path + 'test_cov')
    outputs_ori = readFile(path1 + 'last-hidden')
    
    #tnum_ori = len(cov[0])
    #nnum_ori = len(cov)
    for i in range(len(outputs_ori)):
        outputs_ori[i] = eval(outputs_ori[i])
    '''
    mapdict_ori = {}
    for i in range(nnum_ori):
        mapdict_ori[i] = []
        for j in range(tnum_ori):
            if cov[i][j] == '1':
                mapdict_ori[i].append(j)
            else:
                continue
        mapdict_ori[i] = set(mapdict_ori[i])
    '''
    #print(len(mapdict_ori.keys()))
    #input('check...')
    NSec = getSection(outputs_ori)
    NDist = getDistribute(outputs_ori,NSec)
    NIndex = getSectionIndex(outputs_ori,NSec)
    NMap = getNeuronMap(NIndex)
    #print(NMap)
    #input('check...')
   
    for index in tqdm(range(runN)):
        tt,tkl,tacc = fse(fseter,NDist,outputs_ori,NIndex,NMap)
        #tt = hsg(mapdict_ori,tnum_ori)
        #tt = hsg(testdict,7)
        f = open(path2 + 'fse-'+str(fseter)+ '-' + str(index) + '.result','w')
        f.write(str([tt,tkl,tacc]) + '\n')
        f.close()
        #print('length : %d, %s' %(len(tt),tt))
    '''
    f = open(path + '/' + threshold + '/fse.log','w')
    for item in tlog:
        f.write(str(item) + '\n')
    f.close()
    '''
