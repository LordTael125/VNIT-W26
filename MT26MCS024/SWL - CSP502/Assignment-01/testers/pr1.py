def compute(n):
	sum=0
	k=1
	digit=0
	for i in range(4):
		digit=n*k+digit;
		sum=sum+digit
		k=k*10;
	
	return sum
result1=compute(4)
result2=compute(7)
print(result1)
print(result2)	
