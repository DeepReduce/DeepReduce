import os
import sys
from tqdm import tqdm


def readFile(filepath):
    f = open(filepath)
    content = f.read()
    f.close()
    return content.splitlines()

if __name__ == '__main__':
    threshold = str(sys.argv[2])
    changelayer = int(sys.argv[1])
    path1 = 'doubleneuronnumber-lenet-1/' + str(changelayer) + '/'

    path = path1 + 'Cov/activeneuron/' + threshold + 'ase/'
    cov = readFile(path + 'neuron_cov')
    cnum = len(cov[0])
    nnum = len(cov)
    f = open(path + 'test_cov','w')
    for i in tqdm(range(cnum)):
        tstr = ''
        for j in range(nnum):
            if cov[j][i] == '1':
                tstr += '1'
            else:
                tstr += '0'
        f.write(tstr + '\n')
    f.close()

