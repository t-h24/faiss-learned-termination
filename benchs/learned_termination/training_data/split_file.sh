#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <input>"
    exit 1
fi

input="$1"
output_1="${input}_1"
output_2="${input}_2"

total_lines=$(wc -l < "$input")
midpoint=$((total_lines / 2))

# Split the file in two
head -n "$midpoint" "$input" > "$output_1"
tail -n +"$((midpoint + 1))" "$input" > "$output_2"

echo "Successfully split $input into $output_1 and $output_2."