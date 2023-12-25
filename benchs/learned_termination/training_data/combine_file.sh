#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <input>"
    exit 1
fi

output="$1"
input_1="${output}_1"
input_2="${output}_2"

# Create output file
touch "$output"

# Combine the two files
cat "$input_1" "$input_2" > "$output"

echo "Successfully combined $input_1 and $input_2 into $output."