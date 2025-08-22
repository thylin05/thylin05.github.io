# Input a, b, c
a = int(input())
b = int(input())
c = int(input())

# max a, b, c
def max3(a, b, c):
    vmax = a if a > b else b
    if vmax < c:
        vmax = c
    return vmax
    pass # max3

vmax = max3(a, b, c)

# output
print("Max(%d, %d, %d) = %d"%(a, b, c, vmax))