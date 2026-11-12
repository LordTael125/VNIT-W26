#!/bin/bash

# Program log the user login 

username=$(who)
timestamp=$(date)

directory=~/.cache/USERDATA/SWLB
file="loginrc.txt"
tempfile="loginrc.txt.wr.tmp"

if [ -f "$directory/$file" ]; then
echo "File found : Appending log data"
else
echo "Error: File Not Found : Creating a new log file ..."
mkdir -p "$directory" && touch "$directory/$file"
fi

cp "$directory/$file" "$directory/$tempfile"

echo "User: $(whoami)  logged-in at $(date)" >> "$directory/$tempfile"

cp "$directory/$tempfile" "$directory/$file"

rm "$directory/$tempfile"


