a = 'Category: Automotive(1455 reviews)'

b = a.split('(')[0]
c = a.split('(')[1].replace(' reviews)', '')

print(b)
print(c)