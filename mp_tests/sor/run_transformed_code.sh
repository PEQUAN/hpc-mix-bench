#!/bin/bash

# -------------------------------
# Set CADNA path
# -------------------------------
export load_CADNA_PATH=/home/xinye/.local/lib/python3.12/site-packages/cadnaPromise/cadna

# -------------------------------
# Main folder is current path
# -------------------------------
MAIN_FOLDER="$(pwd)"

# -------------------------------
# Loop over each subfolder
# -------------------------------
for SUBFOLDER in "$MAIN_FOLDER"/*; do
    if [ -d "$SUBFOLDER" ]; then
        echo "Processing folder: $SUBFOLDER"

        # -------------------------------
        # Copy all .mtx files from current folder to subfolder
        # -------------------------------
        cp "$MAIN_FOLDER"/*.mtx "$SUBFOLDER" 2>/dev/null

        # -------------------------------
        # Enter the subfolder
        # -------------------------------
        cd "$SUBFOLDER"


        cp sor.cpp sor_cp.cpp

        # -------------------------------
        # Insert 'using namespace std;' after the last #include
        # -------------------------------
        # Find the last line with #include and insert after it
        awk '/#include/ {last_include=NR} {lines[NR]=$0} END {for (i=1;i<=NR;i++) {print lines[i]; if (i==last_include) print "using namespace std;"}}' sor_cp.cpp > sor_cp_temp.cpp
        mv sor_cp_temp.cpp sor_cp.cpp

        # -------------------------------
        # Compile and run sor_cp.cpp
        # -------------------------------
        g++ sor_cp.cpp -frounding-math -m64 -o sor_cp.out -lcadnaC -L$load_CADNA_PATH/lib -I$load_CADNA_PATH/include
        if [ $? -eq 0 ]; then
            echo "Compilation of sor_cp.cpp successful, running..."
            OUTPUT_CP_FILE="${SUBFOLDER##*/}_cp_result.txt"
            ./sor_cp.out > "$OUTPUT_CP_FILE"
            echo "Output saved to $SUBFOLDER/$OUTPUT_CP_FILE"
        else
            echo "Compilation of sor_cp.cpp failed in $SUBFOLDER"
        fi

        # -------------------------------
        # Remove copied sor_cp.cpp after running
        # -------------------------------
        rm -f sor_cp.cpp sor_cp.out

        # -------------------------------
        # Remove copied .mtx files
        # -------------------------------
        rm -f *.mtx

        # -------------------------------
        # Return to main folder
        # -------------------------------
        cd "$MAIN_FOLDER"
    fi
done

echo "All done!"
