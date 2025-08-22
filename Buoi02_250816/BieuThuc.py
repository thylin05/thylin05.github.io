# Nhập dữ liệu
a = int(input("Moi ban nhap so a: "))
b = int(input("Moi ban nhap so b: "))

# Xử lý
kqCong = a + b
kqTru = a - b
kqNhan = a * b
kqChiaNguyen = a // b
kqDu= a % b
kqChiaThuc = a / b

# Xuất dữ liệu
print("Cac ket qua tinh toan: ")
print("%-5d + %5d = %5d"%(a, b, kqCong))
print("%-5d - %5d = %5d"%(a, b, kqTru))
print("%-5d * %5d = %5d"%(a, b, kqNhan))
print("%-5d / %5d = %5d"%(a, b, kqChiaNguyen))
print("%-5d / %5d = %5d"%(a, b, kqDu))
print("%-5d / %5d = %5.2f"%(a, b, kqChiaThuc))