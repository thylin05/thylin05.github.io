import math

# Nhập dữ liệu
x = float(input("Moi ban nhap vao gia tri cua bien so x: "))

# Xử lý
fx = x + math.pow(x, 5) / (1 * 2 * 3 * 4 * 5) + \
math.sqrt(abs(x)) / math.pow(x, 3.0 / 2)

# Xuất dữ liệu
print("Gia tri cua ham so f(%.2f) = %.2f."%(x, fx))