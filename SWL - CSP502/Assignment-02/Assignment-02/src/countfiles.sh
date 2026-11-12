#!/bin/bash

declare -A ext_counts=([txt]=0 [c]=0 [java]=0 [sh]=0)


while IFS= read -r -d $'\0' file; do

    basename="${file##*/}"

    if [[ "$basename" == *.* ]]; then
        extension="${basename##*.}"

        if [[ -v ext_counts[$extension] ]]; then
            ((ext_counts[$extension]++))
        fi
    fi
done < <(find . -type f -print0)

# Generate the report
echo "================================"
echo "       FILE COUNT REPORT"
echo "================================"
printf "%-10s %s\n" "Extension" "Count"
echo "--------------------------------"

# Print and sort the counts in descending order
for ext in "${!ext_counts[@]}"; do
    printf "%d .%s\n" "${ext_counts[$ext]}" "$ext"
done | sort -rn

echo "================================"
