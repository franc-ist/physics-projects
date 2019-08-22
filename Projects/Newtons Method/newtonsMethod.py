import math

# cos(x) - x^3 = 0
# x_(n+1) = x_n - f(x_n)/[dy/dx]
# abs(x_n) < 1.0e-12

x_old = 0
x_new = 1


def newton(x_old):
    '''This function computes x_(n+1) = x_n - f(x_n)/[dy/dx] '''
    x_new = x_old - ((math.cos(x_old) - math.pow(x_old, 3)) /
                     (-1 * math.sin(x_old) - 3*math.pow(x_old, 2)))
    return x_new


i = 0
while True:
    if abs(x_new - x_old) < 1.0e-12:  # checks if the values have converged
        print('Solution found: {}'.format(x_new))
        break
    else:
        x_old = x_new
        x_new = newton(x_old)
        print('Iteration {}: {}'.format(i, x_new))
        i += 1
