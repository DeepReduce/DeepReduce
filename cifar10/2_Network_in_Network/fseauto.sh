#!/bin/sh

for varible1 in {1..50}
#for varible1 in 1 2 3 4 5
do
    #echo "Hello, Welcome $varible1 times "
    #for varible2 in {100..1000..100}
    #do
    #    python3 fse.py $varible2 $varible1
    #    echo "$varible1-$varible2 : completed!"
    #done
    echo "$varible1 : completed!"
    python3 fse.py 703 $varible1
done
