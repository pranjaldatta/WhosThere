#!/bin/bash
echo $*
if [ -e /home/pranjal/whoconfig.txt ]
    then
    input="/home/pranjal/whoconfig.txt"
    while IFS= read -r line
    do
    data=$line
    done < "$input"

    eval "$(conda shell.bash hook)"
    conda activate facenet
    SCRIPT=$(readlink -f "$0")
    SCRIPT=$(dirname "$SCRIPT")
    result=$(python3 "${SCRIPT}/allow.py" "-$1")   
    if [ $result==1 ]
    then
        if [ $# -gt 1 ]
        then
            echo $data | sudo -S ${@:2} 
        fi    
    else
        echo "Permission Denied."    
    fi
else
    echo "Grant permission first ..."
    read -sp 'Password:' password
    touch /home/pranjal/whoconfig.txt
    echo $password >> /home/pranjal/whoconfig.txt 
fi