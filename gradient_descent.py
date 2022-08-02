import numpy as np

def gradient_descent(x, y, w, b, alpha, compute_gradient, cost_function, iter_num):
    """
    gradient descent

    Args:
        x(ndarray): Shape(m,n), Boston house attributes
        y(ndarray): Shape(m,), Median value of owner-occupied homes in $1000's
        w(ndarray): Coefficient, Parameters of model
        b(float): Intercept, Parameters of model
        alpha(float): learning rate
        compute_gradient: function, compute gradient
        cost_function: function, compute cost
        iter_num(int):
    """
    J_history = []

    for i in range(iter_num):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost = cost_function(x, y, w, b)
        J_history.append(cost)

        # print cost 10 times
        if i % 1000 == 0:
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}")

        if len(J_history) > 2 and np.abs(J_history[-1] - J_history[-2]) < 1e-3:
            break

    return w, b
