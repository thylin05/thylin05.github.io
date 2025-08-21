# Nhập dữ liệu
a = float(input("Moi ban nhap so a: "))
b = float(input("Moi ban nhap so b: "))
# Xử lý
laNhoHon = (a < b)
laNhoHonBang = (a <= b)
laLonHon = (a > b)
laLonHonBang = (a >= b)
laBang = (a == b)
laKhacNhau = (a != b)
# Xuất dữ liệu
print("Ket qua so sanh hai so thuc %.2f va %.2f:"%(a, b))
print("%-8.2f < %8.2f: %8s"%(a, b, laNhoHon))
print("%-8.2f <= %8.2f: %8s"%(a, b, laNhoHonBang))
print("%-8.2f > %8.2f: %8s"%(a, b, laLonHon))
print("%-8.2f >= %8.2f: %8s"%(a, b, laLonHonBang))
print("%-8.2f == %8.2f: %8s"%(a, b, laBang))
print("%-8.2f != %8.2f: %8s"%(a, b, laKhacNhau))
