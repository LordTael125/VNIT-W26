#!/bin/bash

# Script to accept user input and check whether zero, +ve or -ve

read -p "Enter a number : " digit

if [[ "$digit" =~ ^-?[0-9]*\.?[0-9]+$ ]]; then
	echo "The digit $digit has been detected"
else
	echo "Error: $digit is not a valid number"
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


# Checking whether if number is prime or not

if [[ "$digitFlag" =~ "negative" ]]; then
echo "Error: $digit is a negative number primality testing can not be performed"
else
	if [[ "$digit" -eq 1 || "$digit" -eq 2 ]]; then 
		echo "$digit is a prime number"
	else
		for (( i=0; i <

