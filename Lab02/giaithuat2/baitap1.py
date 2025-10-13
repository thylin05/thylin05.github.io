from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

CSV_PATH = "diabetes_prediction_dataset.csv"
FEATURES = ["bmi", "HbA1c_level"]  
TARGET = "diabetes"
C = 1.0   # Regularisation
# Mật độ vẽ vừa phải: giảm số điểm lưới và chỉ vẽ một phần dữ liệu để quan sát
TARGET_GRID_POINTS = 35_000
MAX_SCATTER_POINTS = 8_000  # tối đa số điểm scatter được vẽ (train vẫn dùng full data)

# Load dữ liệu từ CSV
import pandas as pd
df = pd.read_csv(CSV_PATH)
for c in FEATURES + [TARGET]:
    if c not in df.columns:
        raise KeyError(f"Thiếu cột trong CSV: {c}")

data = df[FEATURES + [TARGET]].dropna()
X = data[FEATURES].astype(float).to_numpy()
y = data[TARGET].astype(int).to_numpy()

# Chia train/test 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=109
)

import numpy as np
# LinearSVC thường nhanh hơn cho bài toán tuyến tính khi số mẫu lớn
clf = svm.LinearSVC(C=C, dual=False, max_iter=5000).fit(X_train, y_train)

# Đánh giá nhanh trên tập test và lưu kết quả ra file txt
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
cls_rep = classification_report(
    y_test, y_pred, labels=[0, 1], target_names=["No diabetes", "Diabetes"], digits=4, zero_division=0
)
report_lines = [
    "SVM Linear Results",
    f"CSV: {CSV_PATH}",
    f"Features: {FEATURES}",
    f"Target: {TARGET}",
    f"Kernel: linear, C={C}",
    f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}",
    f"Test accuracy: {acc:.4f}",
    f"Confusion matrix [[TN, FP], [FN, TP]]: {cm.tolist()}",
    "\nClassification report:",
    cls_rep,
]
script_dir = os.path.dirname(__file__)
out_path = os.path.join(script_dir, "svm2d_results.txt")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines) + "\n")
print(f"[SAVE] Đã lưu kết quả SVM vào: {out_path}")

# Tạo lưới để vẽ vùng quyết định
import matplotlib.pyplot as plt

# Phạm vi theo percentile để tránh outlier kéo giãn lưới quá rộng
x_min, x_max = np.percentile(X_train[:, 0], [2, 98])
y_min, y_max = np.percentile(X_train[:, 1], [2, 98])
padx = 0.08 * (x_max - x_min if x_max > x_min else 1.0)
pady = 0.08 * (y_max - y_min if y_max > y_min else 1.0)
x_min, x_max = x_min - padx, x_max + padx
y_min, y_max = y_min - pady, y_max + pady

# Bước lưới thích ứng theo diện tích và mục tiêu số điểm (giới hạn tối thiểu để hình hiển thị thoáng)
area = max((x_max - x_min) * (y_max - y_min), 1e-6)
h = max(0.10, (area / TARGET_GRID_POINTS) ** 0.5)
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h)
)

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(figsize=(6, 4.5))
# pcolormesh thường nhanh hơn contourf với lưới thưa
plt.pcolormesh(xx, yy, Z, shading='auto', cmap=plt.cm.coolwarm, alpha=0.8)
# Chỉ vẽ một phần điểm train để dễ quan sát (train vẫn dùng full data)
plot_idx = np.arange(X_train.shape[0])
if plot_idx.size > MAX_SCATTER_POINTS:
    rng = np.random.default_rng(42)
    plot_idx = rng.choice(plot_idx, size=MAX_SCATTER_POINTS, replace=False)
X_plot = X_train[plot_idx]
y_plot = y_train[plot_idx]

# Vẽ scatter tối ưu render + hiển thị vừa phải
scat = plt.scatter(
    X_plot[:, 0], X_plot[:, 1],
    c=y_plot, s=10, marker='.', edgecolors='none', rasterized=True, alpha=0.8
)
plt.legend(*scat.legend_elements(), loc="upper right", title=TARGET)
plt.xlabel(FEATURES[0])
plt.ylabel(FEATURES[1])
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title(f"Linear SVM — {FEATURES[0]} vs {FEATURES[1]}")
plt.tight_layout()
plt.show()
