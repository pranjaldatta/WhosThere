#!/bin/bash
echo $
DIR="$(cd "$(dirname "$0")" && pwd)"
echo $DIR
eval "$(conda shell.bash hook)"
conda activate facenet
#python3 ./test1.py
A="$(pwd)"
echo $A

echo "2799\$ruban" | sudo -S 