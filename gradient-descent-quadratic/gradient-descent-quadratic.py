def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    ans = x0
    for i in range (steps):
        ans = ans - lr*(2*a*ans + b)
    return ans