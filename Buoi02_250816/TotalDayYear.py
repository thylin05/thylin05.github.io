# Input
y = int(input())

# Process
def isleap(y):
    ans = False
    if y%400==0 or (y%4==0 and y%100 !=0):
        ans = True
    return ans
    pass # IsLeap

def daysyear(y):
    ans = 365
    if isleap(y):
        ans = 366
    return ans
    pass # daysyear

ans = daysyear(y)

# output
print("%d"%(ans))
