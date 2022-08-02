import numpy as np

def cost_function(x, y, w, b):
    """
    Compute cost

    Args:
        x(ndarray): Shape(m,n), Boston house attributes
        y(ndarray): Shape(m,), Median value of owner-occupied homes in $1000's
        w(ndarray): Coefficient, Parameters of model
        b(float): Intercept, Parameters of model
    """
    m = x.shape[0]
    total_loss = 0

    for i in range(m):
        total_loss += (np.dot(w, x[i]) + b - y[i]) ** 2
    total_loss /= (2 * m)

    return total_loss


def compute_gradient(x, y, w, b):
    """
    Compute descent

    Args:
        x(ndarray): Shape(m,n), Boston house attributes
        y(ndarray): Shape(m,), Median value of owner-occupied homes in $1000's
        w(ndarray): Coefficient, Parameters of model
        b(float): Intercept, Parameters of model
    """
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        dj_dw += (np.dot(w, x[i]) + b - y[i]) * x[i]
        dj_db += np.dot(w, x[i]) + b - y[i]
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db
