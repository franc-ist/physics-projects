# (c) 2019 Francis Taylor
# This code is licensed under the MIT license (see LICENSE for details)
# This code is provided as a guide for learning. Please do not copy it blindly.

# m =  [N Σ(xy) − Σx Σy] / [N Σ(x2) − (Σx)2]
# b = [Σy − m Σx ]/ N
# D:\Coding\git\physics-projects\Projects\Least Sqaures\files\LinearTestData.txt

import numpy as np
from numpy import genfromtxt
import pylab as plt
import os

# todo add errors


file_path = input("Enter the path of your file: ")
file_path = file_path.replace('"', '')
if os.path.isfile(file_path):
    values = genfromtxt(file_path, delimiter='\t', dtype=None)
    # print(values)
    x_vals = values[:, 0]
    x_error = values[:, 2]
    y_vals = values[:, 1]
    y_error = values[:, 3]
    A = np.vstack([x_vals, np.ones(len(x_vals))]).T
    m, c = np.linalg.lstsq(A, y_vals, rcond=None)[0]
    print("y={}x + {}".format(m, c))
    plt.plot(x_vals, y_vals, 'o', label='Original data', markersize=3)
    plt.plot(x_vals, m*x_vals + c, 'r', label='Fitted line')
    plt.legend()
    plt.title("y={:f}x + {:f}".format(m, c))
    plt.show()
else:
    print("File does not exist at: {}".format(str(file_path)))
