#!/bin/sh
python3 2process.py 2NBC 0
python3 DeepReduce-coverage.py last-output 2NBC 0
python3 2process.py 3SNAC 0
python3 DeepReduce-coverage.py last-output 3SNAC 0
python3 2process.py 4TKNC_1 0
python3 DeepReduce-coverage.py last-output 4TKNC_1 0
python3 2process.py 4TKNC_2 0
python3 DeepReduce-coverage.py last-output 4TKNC_2 0
python3 2process.py 4TKNC_3 0
python3 DeepReduce-coverage.py last-output 4TKNC_3 0
