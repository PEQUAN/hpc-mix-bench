#!/bin/bash

# colors
RED="\033[0;31m"
GREEN="\033[0;32m"
NOCOLOR="\033[0m"

#
echo "Running some tests"

# iterate over all _cad files
for file in $(ls *_cad)
do
    # if the .txt file is missing, we do not test
    if [ ! -f $file.txt ]; then
        continue
    fi
    
    printf -- "- Checking $file "

    # Compare $file.txt to the output of $file 
    # (remove the line with "CADNA... software")
    # (and used $file.input as input if the file exists)
    if [ -f $file.input ]; then
        diff -q <(grep -v "^CADNA_C.* software$" $file.txt) <(./$file < $file.input | grep -v "^CADNA_C.* software$") &>/dev/null
    else
        diff -q <(grep -v "^CADNA_C.* software$" $file.txt) <(./$file | grep -v "^CADNA_C.* software$") &>/dev/null
    fi
    # display ✓ or ✗
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NOCOLOR}"
    else
        echo -e "${RED}✗${NOCOLOR}"
        echo "-- Comparison (expected result vs current result):"
        # also display the comparison between the two results
        if [ -f $file.input ]; then
            diff -y $file.txt <(./$file < $file.input)
        else
            echo "$file"
            diff -y $file.txt <(./$file)
        fi
        echo ""
    fi
done
