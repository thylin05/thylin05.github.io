# Input a, b, c
a = int(input())
b = int(input())
c = int(input())

# max a, b, c
def max_fn(a, b):
    return a if a > b else b
    pass # max_fn
vmax = max_fn(a, b)
vmax = max_fn(vmax, c)

# output
print("Max(%d, %d, %d) = %d"%(a, b, c, vmax))