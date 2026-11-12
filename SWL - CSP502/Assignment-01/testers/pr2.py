list=[]
for i in range(10):
	a=int(input('Enter value to push:'))
	list.append(a)

for i in range(10):
	a=list.pop()
	print(a)
print(list)	
