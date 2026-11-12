#!/bin/bash

# 1. Store your username in a variable
MY_USER=$(whoami)
# Store formatted current date/time
CURRENT_TIME=$(date "+%Y-%m-%d %H:%M:%S")

# Log file location
LOG_FILE="login_log.txt"

# Grouping all output to print to screen AND append to file at once
{
    echo "========== Login Report: $CURRENT_TIME =========="

    # 2 & 3. Check if user is logged in using 'who'
    # awk extracts the first column, grep -qw checks for exact word match
    if who | awk '{print $1}' | grep -qw "$MY_USER"; then
        echo "User $MY_USER is logged in at $CURRENT_TIME"
    else
        echo "User $MY_USER is not currently logged in."
    fi

    echo "-------------------------------------------------"

    # 5. Count total logged-in users (Counting unique usernames)
    total_unique_users=$(who | awk '{print $1}' | sort -u | wc -l)
    total_sessions=$(who | wc -l)
    echo "Total unique users logged in: $total_unique_users (across $total_sessions sessions)"

    # 6. Log only unique usernames
    echo -e "\nUnique logged-in usernames:"
    who | awk '{print $1}' | sort -u

    # 7. Append your username's logins
    echo -e "\nYour ($MY_USER) active login sessions:"
    who | awk -v user="$MY_USER" '$1 == user {print $0}'

    # 7. Append system-wide logins
    echo -e "\nSystem-wide login information:"
    who

    echo "================================================="
    echo ""
} | tee -a "$LOG_FILE"