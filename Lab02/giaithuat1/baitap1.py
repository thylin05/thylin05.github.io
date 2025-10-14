import numpy as np 
from sklearn import datasets 
from sklearn.model_selection import train_test_split 

# Download&Load dữ liệu iris từ datasets của scikit-learn 
iris = datasets.load_iris() 
# Hiển thị mô ta dữ liệu, chỉ có trong các bộ dữ liệu chuẩn và mở để học tập và nghiên cứu 
print(iris.DESCR) 
# Từ tập dữ liệu ban đầu, tách lấy ma trận biểu diễn các đặc trưng và nhãn
data = iris.data 
target = iris.target 
# Chia dữ liệu và nhãn thành 2 tập dữ liệu huấn luyện và dữ liệu kiểm tra theo tỉ lệ 80:20 
X_train, X_test, y_train, y_test = train_test_split(data, target, 
test_size = 0.2, random_state=101)

from sklearn import svm 
# khởi tạo mô hình phân lớp 
clf = svm.SVC() 
# Sử dụng phương thức 'fit' để huấn luyện mô hình với dữ liệu huấn luyện và nhãn huấn luyện 
# fit (X,Y) với X là tập các đối tượng, Y là tập nhãn tương ứng của đối tượng. 
clf.fit(X_train, y_train) 
# Tính độ chính xác trên tập huấn luyện và tập kiểm tra 
train_acc = clf.score(X_train,y_train) 
val_acc = clf.score(X_test,y_test) 
print('Training accuracy: {}'.format(train_acc)) 
print('Validation accuracy: {}'.format(val_acc))
kernels = ['linear', 'poly', 'rbf', 'sigmoid'] 
best_svm = None 
best_val_acc = -1 
best_kernel = None 
# Huấn luyện các mô hình dựa trên dữ liệu huấn luyện và tham số kernel 
# Tính toán độ chính xác trên tập huấn luyện và tập kiểm tra để tìm được mô hình tốt nhất 
for i in range(4): 
    clf = svm.SVC(kernel=kernels[i], probability=True) 
    clf.fit(X_train, y_train) 
    tmp_val_acc = clf.score(X_test, y_test) 
    if (tmp_val_acc > best_val_acc): 
        best_val_acc = tmp_val_acc 
        best_svm = clf 
        best_kernel = kernels[i] 
# Hiển thị mô hình tốt nhất cùng với độ chính xác 
print("Best validation accuracy : {} with kernel: {}".format(best_val_acc, 
best_kernel))     