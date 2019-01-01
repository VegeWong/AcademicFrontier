#!/bin/bash

data_dir="/home/bcmilht/datasets/ILSVRC/Data/CLS-LOC/train/"
target_dir="/home/wanghanjing/AcademicFrontier/data/"
num_class=10
cnt=0
for folder in $data_dir*; do
    cnt=$(($cnt+1))
    echo "cnt= " $cnt
    echo "folder= " $folder
    cp -r $folder $target_dir
    if (( $cnt == $num_class )); then
        break
    fi
done
