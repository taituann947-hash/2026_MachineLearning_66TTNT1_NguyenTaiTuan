import numpy as np
def grad1(x):
    return 2 * x

def cost1(x):
    return x**2 - 2

def myGD1_bai1(x0, eta):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta * grad1(x[-1])
        if abs(grad1(x_new)) < 1e-3:
            break
        x.append(x_new)
    return (x, it)
(x1_b1, it1_b1) = myGD1_bai1(-5, 0.1)
(x2_b1, it2_b1) = myGD1_bai1(5, 0.1)

print("--- KẾT QUẢ BÀI 1 ---")
print('Solution x1 = %f, cost = %f, after %d iterations' % (x1_b1[-1], cost1(x1_b1[-1]), it1_b1))
print('Solution x2 = %f, cost = %f, after %d iterations' % (x2_b1[-1], cost1(x2_b1[-1]), it2_b1))
