# Input a, b with ax + b = 0
a = float(input())
b = int(input())

# Process
def solv1deg(a, b):
    """
    Solve Equation ax + b = 0
    """
    nsol, x = None, None
    if a == 0:
        if b == 0:
            nsol = -1
        else:
            nsol = 0
    else:
        nsol = 1
        x = -b / a

    return nsol, x
    pass # solv1deg

nsol, x = solv1deg(a, b)

# output
if nsol == -1:
    print("%.2fx + %.2f = 0: unlimited solutions"%(a, b))
elif nsol == 0:
    print("%.2fx + %.2f = 0: no solution"%(a, b))
else:
    print("%.2fx + %.2f = 0: x = %.2f"%(a, b, x))