#!/bin/bash
if [ -e /etc/whoconfig.txt ]
    then
    input="/etc/whoconfig.txt"
    while IFS= read -r line
    do
    data=$line
    done < "$input"

    eval "$(conda shell.bash hook)"
    conda activate facenet
    SCRIPT=$(readlink -f "$0")
    SCRIPT=$(dirname "$SCRIPT")
    result=$(python3 "${SCRIPT}/allow.py" "-$1") 
    if [ ${#result} -gt 1 ]
    then 
        result=200
    fi    
    
    if [ $result == 1 ]
    then   
        if [ $# -gt 1 ]
        then
            echo $data | sudo -S ${@:2} 
        else
            echo "Successfully verified."   
        fi    
    elif [ $result==200 ]
    then    
        echo Success.
    else
        echo "Permission Denied."    
    fi
else
    echo "Grant permission first ..."
    read -sp 'Password:' password
    echo $password | sudo -S touch /etc/whoconfig.txt   
    sudo chmod 666 /etc/whoconfig.txt
    echo $password | sudo -S echo $password >> /etc/whoconfig.txt 
fi