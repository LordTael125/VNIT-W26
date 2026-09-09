#!/bin/bash

# Script to accept user input and check whether zero, +ve or -ve

read -p "Enter a number : " digit

# Check whether input is a valid integer
if [[ "$digit" =~ ^-?[0-9]+$ ]]; then

    echo "The digit $digit has been detected"

else

    echo "Error: $digit is not a valid number"
    exit 1

fi


# Checking number flag

if [[ "$digit" -lt 0 ]]; then

    echo "The digit $digit is a negative number"
    digitFlag="negative"

elif [[ "$digit" -gt 0 ]]; then

    echo "The digit $digit is a positive number"
    digitFlag="positive"

else

    echo "The digit is zero"
    digitFlag="zero"

fi


# Checking whether number is prime or not

if [[ "$digitFlag" == "negative" || "$digitFlag" == "zero" ]]; then

    echo "Prime checking cannot be performed for $digit"

elif [[ "$digit" -eq 1 ]]; then

    echo "$digit is not a prime number"

elif [[ "$digit" -eq 2 ]]; then

    echo "$digit is a prime number"

else

    is_prime=1

    for (( i=2; i*i<=digit; i++ ))
    do
        if (( digit % i == 0 )); then
            is_prime=0
            break
        fi
    done

    if [[ "$is_prime" -eq 1 ]]; then
        echo "$digit is a prime number"
    elsechmod +x numcheck.sh
        echo "$digit is not a prime number"
    fi

fi