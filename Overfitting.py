
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1234)


# Các hàm

def poly(x, d):
    return np.hstack([x**i for i in range(d + 1)])


def train(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y


def predict(X, w):
    return X @ w


def mse(y, pred):
    return np.mean((y - pred) ** 2)


# Tạo dữ liệu

n = 12

# Dữ liệu đầu vào
x = np.sort(
    np.random.rand(n, 1) * 4,
    axis=0
)

# Dữ liệu có nhiễu
y = (
    np.sin(x).ravel()
    + np.random.normal(0, 0.25, n)
)

# Dữ liệu để kiểm tra và vẽ đường cong
x_test = np.linspace(
    0, 4, 100
).reshape(-1, 1)

y_test = np.sin(
    x_test
).ravel()


# Kiểm tra Overfitting

degree_overfit = 10

X_overfit = poly(
    x,
    degree_overfit
)

X_test_overfit = poly(
    x_test,
    degree_overfit
)

# Train mô hình Degree 10
w_overfit = train(
    X_overfit,
    y
)

# Dự đoán
train_pred = predict(
    X_overfit,
    w_overfit
)

test_pred = predict(
    X_test_overfit,
    w_overfit
)

# Tính MSE
train_mse = mse(
    y,
    train_pred
)

test_mse = mse(
    y_test,
    test_pred
)

print("OVERFITTING")

print(
    "Degree:",
    degree_overfit
)

print(
    "Train MSE:",
    round(train_mse, 4)
)

print(
    "Test MSE:",
    round(test_mse, 4)
)


# K-fold Cross Validation 

k = 4

degrees = range(1, 11)

errors = []


# Chia Fold CHỈ MỘT LẦN
folds = np.array_split(
    np.random.permutation(n),
    k
)


# Thử từng Degree
for d in degrees:

    fold_errors = []

    # K lần train/validation
    for i in range(k):

        # Validation
        val = folds[i]

        # Training = các fold còn lại
        train_idx = np.hstack(
            [
                folds[j]
                for j in range(k)
                if j != i
            ]
        )

        # Tạo Polynomial Features
        X_train = poly(
            x[train_idx],
            d
        )

        X_val = poly(
            x[val],
            d
        )

        # Train
        w = train(
            X_train,
            y[train_idx]
        )

        # Predict
        val_pred = predict(
            X_val,
            w
        )

        # MSE
        error = mse(
            y[val],
            val_pred
        )

        fold_errors.append(
            error
        )

    # MSE trung bình của Degree này
    errors.append(
        np.mean(fold_errors)
    )


# Chọn Degree tốt nhất 

best_d = list(degrees)[
    np.argmin(errors)
]

print("\nK-FOLD ")

print(
    "Best Degree:",
    best_d
)


# Train lại với Best Degree 

X_best = poly(
    x,
    best_d
)

X_test_best = poly(
    x_test,
    best_d
)

# Train lại bằng TOÀN BỘ dữ liệu
w_best = train(
    X_best,
    y
)

# Dự đoán
best_pred = predict(
    X_test_best,
    w_best
)

# Test MSE
final_test_mse = mse(
    y_test,
    best_pred
)

print(
    "Final Test MSE:",
    round(final_test_mse, 4)
)


# Vẽ kết quả 

plt.figure(
    figsize=(10, 6)
)


# Dữ liệu training
plt.scatter(
    x,
    y,
    color="black",
    label="Training data"
)


# Đường sin thật
plt.plot(
    x_test,
    y_test,
    "k--",
    alpha=0.5,
    label="sin(x)"
)


# Mô hình Overfit - Degree 10
plt.plot(
    x_test,
    test_pred,
    color="red",
    label="Overfit (Degree 10)"
)


# Mô hình tốt nhất
plt.plot(
    x_test,
    best_pred,
    color="green",
    linewidth=2.5,
    label=f"Best model (Degree {best_d})"
)


plt.xlabel("x")
plt.ylabel("y")

plt.title(
    "Overfitting and K-fold Cross Validation"
)

plt.legend()

plt.show()


# Dự đoán giá trị mới 

a = float(
    input("Input x: ")
)


# Tạo Polynomial Features
X_new_best = poly(
    np.array([[a]]),
    best_d
)

X_new_overfit = poly(
    np.array([[a]]),
    degree_overfit
)


# Dự đoán bằng Best Model
result_best = predict(
    X_new_best,
    w_best
)[0]


# Dự đoán bằng Overfit Model
result_overfit = predict(
    X_new_overfit,
    w_overfit
)[0]


# Giá trị thật
real = np.sin(a)


print("\nPREDICTION ")

print(
    "Overfit prediction:",
    round(result_overfit, 4)
)

print(
    "Best model prediction:",
    round(result_best, 4)
)

print(
    "Real value:",
    round(real, 4)
)
