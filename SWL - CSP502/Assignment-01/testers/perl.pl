#!/usr/bin/perl
use strict;
use warnings;
print "HELLO\n";
print "Enter first number :";
my $num1 = <STDIN>;
chomp $num1;
print "Enter second number :";
my $num2 = <STDIN>;
chomp $num2;
my $sum = $num1 + $num2;
print "The sum is : $sum\n";

@nums=(10,45,23,89,12);
$max = $nums[0];
foreach $n (@nums)
{
	if($n>$max)
	{
		$max=$n;
	}
}
print "$max\n";


