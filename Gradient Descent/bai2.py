import numpy as np
def grad2(x):
    return x**2 - 1

def cost2(x):
    return (1/3) * (x**3) - x

def myGD1_bai2(x0, eta):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta * grad2(x[-1])
        if abs(grad2(x_new)) < 1e-3:
            break
        x.append(x_new)
    return (x, it)
(x1_b2, it1_b2) = myGD1_bai2(0.5, 0.1)
(x2_b2, it2_b2) = myGD1_bai2(2.0, 0.1)

print("\n--- KẾT QUẢ BÀI 2 ---")
print('Solution x1 = %f, cost = %f, after %d iterations' % (x1_b2[-1], cost2(x1_b2[-1]), it1_b2))
print('Solution x2 = %f, cost = %f, after %d iterations' % (x2_b2[-1], cost2(x2_b2[-1]), it2_b2))
