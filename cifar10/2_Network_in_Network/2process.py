import os
import sys
from tqdm import tqdm


def readFile(filepath):
    f = open(filepath)
    content = f.read()
    f.close()
    return content.splitlines()

if __name__ == '__main__':
    # threshold = str(sys.argv[1])
    filename = str(sys.argv[1])
    # path = os.getcwd() + '/Cov/activeneuron/' + threshold + 'ase/'
    path = os.getcwd() + '/Cov/'
    # cov = readFile(path + 'neuron_cov')
    cov = readFile(path + filename)
    cnum = len(cov[0])
    nnum = len(cov)
    # f = open(path + 'test_cov','w')
    f = open(path + 'test_cov_' + filename, 'w')
    for i in tqdm(range(cnum)):
        tstr = ''
        for j in range(nnum):
            if cov[j][i] == '1':
                tstr += '1'
            else:
                tstr += '0'
        f.write(tstr + '\n')
    f.close()

