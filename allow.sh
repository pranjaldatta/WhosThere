#!/bin/bash

if [ -e /home/pranjal/whoconfig.txt ]
    then
    input="/home/pranjal/whoconfig.txt"
    while IFS= read -r line
    do
    data=$line
    done < "$input"

    eval "$(conda shell.bash hook)"
    conda activate facenet
    result=$(python3 /home/pranjal/Projects/WhosThere/allow.py "-$1")   
  
    if [ $result==1 ]
    then
        echo $data | sudo -S ${@:2} 
    else
        echo "Permission Denied."    
    fi
else
    echo "Grant permission first ..."
    read -sp 'Password:' password
    touch /home/pranjal/whoconfig.txt
    echo $password >> /home/pranjal/whoconfig.txt 
fi