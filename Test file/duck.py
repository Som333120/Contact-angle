import math
x = int(20)
y = int(3)
h = float(1)

r = x/2
area =int((math.pi)*(r**2))
areaa = int(area*y)
time = math.ceil((areaa*h))
day = math.ceil(time/24)
print("Pool volume = ",areaa ,"cubic meter")
print("Minimal time to use = ",time ,"hour")
print("Minimal day to use = ",day ,"day")