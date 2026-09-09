#!/bin/bash

declare -i hiddenFiles visibleFiles totalEntries visibleEntries visibleDirectories hiddenDirectories hiddenEntries

totalEntries=$(ls -1A | wc -l)
#Debug 01
echo "Total Entries Counted = $totalEntries"

visibleEntries=$(ls -1 | wc -l)

visibleDirectories=$(ls -1 -d */ | wc -l)
#Debug 02

hiddenEntries=$(find . -mindepth 1 -maxdepth 1 -name ".*" | wc -l)
#Debug 03

hiddenDirectories=$(find . -mindepth 1 -maxdepth 1 -type d -name ".*" | wc -l)
#Debug 04

#Arthmetic derviation of count values
hiddenFiles=$((hiddenEntries - hiddenDirectories))
visibleFiles=$((visibleEntries - visibleDirectories))

echo "Visible Entries found = $visibleEntries"

echo "Visible Files found = $visisbleFiles"

echo "Visible Directories found = $visibleDirectories"

echo "Hidden Entries found = $hiddenEntries"

echo "Hidden Files found = $hiddenfiles"

echo "Hidden Directories found = $hiddenDirectories"
