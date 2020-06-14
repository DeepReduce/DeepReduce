import os
import copy
import random
import math
import numpy as np
from collections import defaultdict
import numpy as np
import sys
import gc

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
        y = model1[RST[i]][0]
        if y == 1:
            acc += 1
        else:
            pass
        count += 1
    return acc/count

global KN
KN = 20


def update_dist(RS,RSTDist,outputs,NIndex):
    for i in RS:
        output = NIndex[i]
        for j in range(len(RSTDist)):
            tempindex = output[j]
            RSTDist[j][tempindex] += 1
    return RSTDist

# ds : distribution of sample set; dt : distribution of test set
def calculate_kl(ds,dt,snum):
    #klsum = 0
    eps = 1e-15
    if snum == 0:
        #eps = 1e-15
        klsum = 0
        #print(len(ds))
        for i in range(len(ds)):
            for k in range(KN):
                klsum += max(dt[i][k],eps) * np.log(max(dt[i][k],eps)/eps)
    else:
        klsum = 0
        for i in range(len(ds)):
            for k in range(KN):
                klsum += max(dt[i][k],eps) * np.log(max(dt[i][k],eps)/max(ds[i][k]/snum,eps))
    return klsum/len(ds)

def ckl(ds,dt):
    eps = 1e-15
    cesum = 0
    for i in range(len(ds)):
        for k in range(KN):
            cesum += max(dt[i][k],eps) * np.log(max(dt[i][k],eps)/max(ds[i][k],eps)) 
    return cesum/len(ds)


def SelectTest_dlr(SIZE,LIST,mapdict,curdict,tnum,marked,MAX_CARD,maxdepth):
    count = [0]*tnum
    maxcount = 0
    #if SIZE not in curdict.keys():
    #    return SelectTest(SIZE+1,LIST,mapdict,curdict,tnum,marked,MAX_CARD)
    for ti in LIST:
        temp = 0
        #try:
        for ni in curdict[SIZE]:
            if (marked[ni] == False) and (ti in mapdict[ni]):
                temp += 1
        count[ti]
        try:
            count[ti] = temp
        except:
            print('error ...')
            print('length : %d : %s'%(len(count),count))
            print(ti)
            input('SelectTest check...')
        maxcount = max(maxcount,count[ti])
    testlist = []
    for ti in LIST:
        if count[ti] == maxcount:
            testlist.append(ti)
    if len(testlist) == 1:
        return testlist[0]
    elif SIZE == MAX_CARD:
        try:
            return testlist[random.randint(0,len(testlist)-1)]
        except:
            print('error2 ...')
            print(LIST)
            print(testlist)
            input('check...')
    else:
        #print('debug')
        cc = list(curdict.keys())
        cc.sort()
        ccindex = cc.index(SIZE)
        if maxdepth != 0:
            maxdepth = maxdepth - 1
            return SelectTest_dlr(cc[ccindex+1],testlist,mapdict,curdict,tnum,marked,MAX_CARD,maxdepth)
        else:
            return testlist[random.randint(0,len(testlist)-1)]


def DR_kl(mapdict,tnum,NDist,outputs,NIndex,NMap,filepath):

    f = open(filepath,'w')
    RST = []
    log = []
    dnum = len(NDist)
    RSTDist = []
    for i in range(dnum):
        RSTDist.append([0] * KN)
    terminal_cov = 0
    terminal_ce =  float('inf')
   

    # first iteration to guarantee coverage for reduction

    for first_iteration in range(1):
        #mapdict,tnum = update(mapdict,tnum,RTT)
        # initialization
        marked = [False] * len(mapdict.keys())
        MAX_CARD = 0
        RS = []
        for i in mapdict.keys():
            MAX_CARD = max(MAX_CARD,len(mapdict[i]))
            if len(mapdict[i]) == 1:
                #RS.append(mapdict[i][0])
                if list(mapdict[i])[0] not in RS:
                    RS.extend(list(mapdict[i]))
                marked[i] = True
        for i in mapdict.keys():
            if mapdict[i] & set(RS):
                marked[i] = True
        cur_card = 1

        curdict = {}
        for i in mapdict.keys():
            tlen = len(mapdict[i])
            if tlen not in curdict.keys():
                curdict[tlen] = [i]
            else:
                curdict[tlen].append(i)
        #print('initiation : %s'%RS)

        while MAX_CARD > cur_card:
        #List = []
            cur_card += 1
            if cur_card not in curdict.keys():
                #print('%s not exists ...'%cur_card)
                continue
            while True:
                if cur_card not in curdict.keys():
                    break
                #print('size : %s '%cur_card)
                List = []
                for item in curdict[cur_card]:
                    if marked[item] == False:
                        List.extend(mapdict[item])
                List = list(set(List))
                if len(List)== 0:
                    break
                next_test = SelectTest_dlr(cur_card,List,mapdict,curdict,tnum,marked,MAX_CARD,3)
                #print('*************************')
                #print('list : %s'%List)
                #print('candidate list : %s , candidate test : %s' %(List,next_test))
                RS.append(next_test)
                may_reduce = False
                for i in mapdict.keys():
                    if next_test in mapdict[i]:
                        marked[i] = True
                        if len(mapdict[i]) == MAX_CARD:
                            may_reduce = False
                    else:
                        continue
                if may_reduce:
                    for i in mapdict.keys():
                        if marked[i] == False:
                            MAX_CARD = max(MAX_CARD,len(mapdict[i]))
                #print('marked : %s'%marked)
                #print('cur_card : %d, reduced test : %s'%(cur_card,RS))
                #print('max_card : %d, cur_card : %d'%(MAX_CARD,cur_card))

                #input('check...')
            #input('cry...')
        RST.extend(copy.deepcopy(RS))
        RST = list(set(RST))
        RS = list(set(RS))
        RSTDist = update_dist(RS,RSTDist,outputs,NIndex)
        #terminal_cov = getCoverage(mapdict,RST)
        terminal_cov = '-'
        terminal_ce = calculate_kl(RSTDist,NDist,len(RST))
        #terminal_acc = calculate_acc(predict,RST)
        terminal_acc = calculate_acc(predict,RST)
        print('reduced test : %s, coverage : %s, cross entropy : %s, accuracy: %s'%(len(RST),terminal_cov,terminal_ce,terminal_acc))
        log.append('reduced test : %s, coverage : %s, cross entropy : %s, accuracy : %s'%(len(RST),terminal_cov,terminal_ce,terminal_acc))
        f.write(str(RST) + '\n')
        #input('iteration check...')

    # release memory
    del mapdict
    gc.collect()
 
    # the following iteration to guarantee test distribution
    print('starting second iteration ...')
    #RSTR = copy.deepcopy(RST)
    #RSTRDist = copy.deepcopy(RSTDist)
    #flag = 1
    while len(RST) < 1000:
        RS = []
        #RS = getCandidate(RSTDist,NDist,NMap,mapdict,len(RST),RST)
        RS = getCandidate_new(RSTDist,NDist,NMap,len(RST),RST,NIndex,outputs)
        RSTDist = update_dist(RS,RSTDist,outputs,NIndex)
        #mapdict,tt2,RSTRDist = update_dist(mapdict,tnum,RSTR,RSR,RSTRDist,outputs,NIndex,NMap)
        RST.extend(RS)
        #RSTR.extend(RSR)
        #terminal_cov = getCoverage(mapdict,RST)
        terminal_ce = calculate_kl(RSTDist,NDist,len(RST))
        #terminal_acc = '-'
        terminal_acc = calculate_acc(predict,RST)
        #terminal_cer = calculate_cross_entropy(RSTRDist,NDist,len(RSTR))
        #terminal_accr = calculate_acc(predict,RSTR)
        print('reduced test : %s, kl : %s, accuracy: %s'%(len(RST),terminal_ce,terminal_acc))
        #print('random test : %s, coverage : %s, cross entropy : %s, accuracy: %s'%(len(RSTR),terminal_cov,terminal_cer,terminal_accr))
        print('************************************************************************************************')
        log.append('reduced test : %s, kl : %s, accuracy : %s'%(len(RST),terminal_ce,terminal_acc))
        f.write(str(RST) + '\n')
        #input('iteration check...')
    f.close()
    return RST,log


def getCandidate_new(RSTDist,NDist,NMap,tnum,RST,NIndex,outputs):
    eps = 1e-15
    relist = []
    for i in range(len(RSTDist)):
        maxindex = (i,0)
        maxdiff = NDist[i][0]*tnum/max(RSTDist[i][0],eps)
        for j in range(len(RSTDist[i])):
            tempdiff = NDist[i][j]*tnum/max(RSTDist[i][j],eps)
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
    minkl = float('inf')
    minindex = -1
    for i in range(len(simdict[maxsim])):
        exp = copy.deepcopy(RST)
        exp.append(simdict[maxsim][i])
        exdist =  update_dist([simdict[maxsim][i]],copy.deepcopy(RSTDist),outputs,NIndex)
        tempkl = calculate_kl(exdist,NDist,len(exp))
        if tempkl < minkl:
            minkl = tempkl
            minindex = i
    print('length of candidate : %s, maxsimilarity : %s'%(len(simdict[maxsim]),maxsim))
    #return random.sample(simdict[maxsim],1)
    return [simdict[maxsim][minindex]]

 

def calSimilarity(a,b):
    resim = 0
    for i in range(len(a)):
        if a[i] == b[i]:
            resim += 1
    return resim 



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

def getSection_heavytailed_old(outputs):
    nnum = len(outputs[0])
    tnum = len(outputs)
    KIndex = []
    for i in range(tnum):
        KIndex.append([0]*nnum)
        
    for n in range(nnum):
        templist = []
        tempindex = [0]*KN
        for i in range(tnum):
            templist.append(outputs[i][n])
        templist.sort()
        for i in range(KN):
            tempindex[i] = templist[int((tnum/KN)*i)]
        if n == 1:
            print(templist)
            print(tempindex)
            input('check...')
        for i in range(tnum):
            for j in range(KN):
                if outputs[i][n] < tempindex[j]:
                    KIndex[i][n] = j
                    break
                else:
                    continue
            #print(outputs[i][n])
            #print(KIndex[i][n])
            #print(tempindex)
            #input('check...')
    return KIndex

def getSection_heavytailed(outputs):
    nnum = len(outputs[0])
    tnum = len(outputs)
    KIndex = []
    for i in range(tnum):
        KIndex.append([0]*nnum)

    for n in range(nnum):
        templist = []
        tempindex = [0]*KN
        for i in range(tnum):
            templist.append(outputs[i][n])
        templist = list(set(templist))
        templist.sort()
        for i in range(KN):
            tempindex[i] = templist[int((len(templist)/KN)*i)]
        for i in range(tnum):
            for j in range(KN):
                if j < KN-1:
                    if tempindex[j] <= outputs[i][n] and outputs[i][n] < tempindex[j+1]:
                        KIndex[i][n] = j
                        break
                elif j == KN-1:
                    if tempindex[j] <= outputs[i][n]:
                        KIndex[i][n] = j
                        break
        #if n == 1:
        #    print(templist)
        #    print(tempindex)
        #    input('check...')
    return KIndex

def getDistribute_heavytailed(NIndex):
    NDist = []
    for n in range(len(NIndex[0])):
        NDist.append([0]*KN)
    for i in range(len(NIndex)):
        for n in range(len(NIndex[i])):
            NDist[n][NIndex[i][n]] += 1
    for n in range(len(NIndex[i])):
        for k in range(KN):
            NDist[n][k] = NDist[n][k]/len(NIndex)
    return NDist


if __name__ == "__main__":
    input_file = str(sys.argv[1])
    # threshold = str(sys.argv[2])
    filename = str(sys.argv[2])
    if(filename[0] == '0'):
        path =  path = os.getcwd() + '/input/'
        path_res = os.getcwd() + '/result/DeepReduce/' + filename + '/'
    else:
        path = os.getcwd() + '/input/test_cov_'
        path_res = os.getcwd() + '/result/DeepReduce/' + filename + '/'
    if os.path.exists(path_res) == False:
        os.makedirs(path_res)
    if (filename[0] == '0'):
        cov = readFile(path + 'Cov/activeneuron/'+  filename + 'ase/test_cov')
    else:
        cov = readFile(path + filename)
    outputs_ori = readFile(os.getcwd() + '/input/' + input_file)
    tnum_ori = len(cov[0])
    nnum_ori = len(cov)
    for i in range(len(outputs_ori)):
        outputs_ori[i] = eval(outputs_ori[i])
    mapdict_ori = {}
    for i in range(nnum_ori):
        mapdict_ori[i] = []
        for j in range(tnum_ori):
            if cov[i][j] == '1':
                mapdict_ori[i].append(j)
            else:
                continue
        mapdict_ori[i] = set(mapdict_ori[i])
    del cov
    gc.collect()
    #input('memory check...')
    #NSec = getSection(outputs_ori)
    #NDist = getDistribute(outputs_ori,NSec)
    #NIndex = getSectionIndex(outputs_ori,NSec)
    NIndex = getSection_heavytailed(outputs_ori)
    NDist = getDistribute_heavytailed(NIndex)
    NMap = getNeuronMap(NIndex)
    for i in NDist:
        print(i)
    #print(NMap)
    #input('check...')
    
    #tt,tlog = DR_kl(NDist,outputs_ori,NIndex,NMap,path_res + 'DR-%s-%s.iteration'%(KN,input_file))
    tt,tlog = DR_kl(mapdict_ori,tnum_ori,NDist,outputs_ori,NIndex,NMap,path_res + 'DRcov-%s-%s.iteration.tt'%(KN,input_file))
    #tt = hsg(mapdict_ori,tnum_ori)
    #tt = hsg(testdict,7)
    f = open(path_res + 'DRcov-%s-%s.result.tt'%(KN,input_file),'w')
    f.write(str(tt) + '\n')
    f.close()
    print('length : %d, %s' %(len(tt),tt))
    
    f = open(path_res + 'DRcov-%s-%s.log.tt'%(KN,input_file),'w')
    for item in tlog:
        f.write(str(item) + '\n')
    f.close()
