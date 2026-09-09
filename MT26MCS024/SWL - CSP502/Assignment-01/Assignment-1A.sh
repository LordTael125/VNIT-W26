#!/bin/bash

TARGET_DIR="${1:-.}"

if [[ ! -d "$TARGET_DIR" ]]; then
    echo "Error: '$TARGET_DIR' is not a valid directory." >&2
    exit 1
fi

file_count=$(find "$TARGET_DIR" -maxdepth 1 -type f -printf '.' | wc -c)

echo "Total files: $file_count"





















