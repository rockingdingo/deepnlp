#!/bin/bash

FOLDER_TEST=`pwd`
echo "NOTICE: Testing Folder is "${FOLDER_TEST}

echo "NOTICE: Testing python2 function"
for file in ${FOLDER_TEST}/*; do  
    if [ ${file##*.} = "py" ]; then  # Get all python script
        echo "NOTICE: Python2 "${file}
        python ${file}
    fi
done

echo "NOTICE: Testing python3 function"
for file in ${FOLDER_TEST}/*; do  
    if [ ${file##*.} = "py" ]; then  # Get all python script
        echo "NOTICE: Python3 "${file}
        python3 ${file}
    fi
done
