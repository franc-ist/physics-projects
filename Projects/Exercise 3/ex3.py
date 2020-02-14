#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author:  Francis Taylor
# Most comments are above the line they are commenting

import cmath
# import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# define constants
epsilon = 8.85E-12
c = 299792458


def f(k, z, x, x_prime):
    '''
    Evaluates the complex exponential part of the Fresnel diffraction function
    within the integral.


    Parameters
    ----------
        k : float
            Wavenumber of the light.
        z : float
            Distance from the aperture to the screen.
        x : float
            Coordinate on the screen.
        x_prime : float
            Coordinate on the aperture.

    Returns
    -------
        f : float
            The evaluated complex exponential.
    '''
    return np.exp(1j*k/(2*z) * np.square(x-x_prime))


def simpson_integral(x_prime, k, x, z, N):
    '''
    Performs a Simpson's rule numerical integration on a given function. For
    each value x, the function iterates over all x' values between x'1 and x'2.


    Parameters
    ----------
        x_prime : array
            The range of x' to integrate over.
        k : float
            Wavenumber of the light.
        x : array
            The range of x on the screen to iterate over.
        z : float
            The distance to the screen.
        N : int
            The number of terms to iterate over.

    Returns
    -------
        I : array
            Computed relative intensity values for the diffraction.
    '''
    # note: for each value x, integrate over range x'1 to x'2
    # extension: compare against scipy.integrate.simps()

    # generate arrays for E and I
    E = np.zeros(N, dtype=np.complex_)  # allows handling of complex values
    I = np.zeros(N)

    # iterate over all screen coordinates
    for i1 in range(len(x)):

        # define the spacing
        h = abs(x2_prime - x1_prime)/N

        # generate a temparory sum array
        sum = np.zeros(N, dtype=np.complex_)

        # assign values for x at x'1 and x'2
        sum[0] = (h/3) * f(k, z, x1, x1_prime)
        sum[N-1] = (h/3) * f(k, z, x2, x2_prime)

        # integrate over all x'
        for i2 in range(len(x_prime)):
            # evaluate even values of N
            if i2 % 2 == 0:
                sum[i2] = h/3 * 2 * f(k, z, x[i1], x_prime[i2])
            # evaluate odd values of N
            else:
                sum[i2] = h/3 * 4 * f(k, z, x[i1], x_prime[i2])

        # fix for constants & convert to intensity
        E[i1] = np.sum(sum) * (k*E0/2*np.pi*z)
        I[i1] = np.real(epsilon * c * E[i1] * np.conjugate(E[i1]))
    return I


user_input = '0'
while user_input != 'q':
    user_input = input(
        '\nChoose an option:\n'
        '\na: 1-D integration,'
        '\nb: 1-D integration with custom values,'
        '\nc: 2-D integration.'
        '').lower()

    if user_input == 'a':
        print('Performing a 1-dimensional Fresnel integral uing Simpson\'s '
              'rule, with default values.')
        _lambda = 1E-6
        k = 2*np.pi/_lambda
        N = 100
        E0 = 1

        x1 = -0.005
        x2 = 0.005
        x1_prime = -1E-5
        x2_prime = 1E-5
        z = 0.02

        x = np.linspace(x1, x2, N)
        x_prime = np.linspace(x1_prime, x2_prime, N)

        E = simpson_integral(x_prime, k, x, z)

        plt.plot(x, E)
        plt.xlabel('Screen coordinate, x (m)', size=12)
        plt.ylabel('Relative intensity', size=12)
        plt.plot()

    if user_input == 'b':
        print('Performing a 1-dimensional Fresnel integral uing Simpson\'s '
              'rule, with custom values. Leave blank for the default value.')
        _lambda = float(input(
            'Please enter a value for lambda (in metres - accepts scientific s'
            'notation): ') or 1E-6)
        k = 2*np.pi/_lambda
        N = int(input('Please enter a value for N:') or 100)
        E0 = float(
            input('Please enter a value for the initial electrical field (E0):'
                  ) or 1)

        x1 = float(input(
            'Please enter a value for the minimum x coordinate (in metres - '
            'accepts scientific notation):') or -0.005)
        x2 = float(input(
            'Please enter a value for the maximum x coordinate (in metres - '
            'accepts scientific notation):') or 0.005)
        x1_prime = float(input(
            'Please enter a value for the minimum aperture limit (in metres - '
            'accepts scientific notation):') or -1E-5)
        x2_prime = float(input(
            'Please enter a value for the maximum aperture limit (in metres - '
            'accepts scientific notation):') or 1E-5)
        z = float(input('Please enter a value for the distance between the '
                        'aperture and the screen (in metres - accepts '
                        'scientific notation): ')
                  or 0.02)

        x = np.linspace(x1, x2, N)
        x_prime = np.linspace(x1_prime, x2_prime, N)

        E = simpson_integral(x_prime, k, x, z)

        plt.plot(x, E)
        plt.xlabel('Screen coordinate, x (m)', size=12)
        plt.ylabel('Relative intensity', size=12)
        plt.plot()

    if user_input == 'c':
        print('Performing a 2-dimensional Fresnel integral uing Simpson\'s '
              'rule, with custom values. Leave blank for the default value.')
        # loop for each value until valid input
        while True:
            try:
                _lambda = float(input(
                    'Please enter a value for lambda (in metres - accepts '
                    'scientific notation): ') or 1E-6)
                if (_lambda <= 0):
                    raise ValueError('The wavelength must be positive and '
                                     'greater than 0. Please try again.')

            except ValueError:
                print('Invalid input. Please try again.')
                continue
            else:
                break
        while True:
            try:
                N = int(input('Please enter a value for N:') or 100)
                if (N <= 0):
                    raise ValueError(
                        'N must be larger than 0. Please try again.')
                elif (N > 500):
                    raise ValueError(
                        'The value of N entered is too large, and will cause '
                        'performance degradation. Please select a value lower '
                        'than 500.')
            except ValueError:
                print('Invalid input. Please try again.')
                continue
            else:
                break
        while True:
            try:
                E0 = float(input('Please enter a value for the initial '
                                 'electrical field (E0):') or 1)
                if (E0 < 0):
                    raise ValueError(
                        'E0 cannot be negative. Please enter a positive '
                        'value.')
            except ValueError:
                print('Invalid input. Please try again.')
                continue
            else:
                break
        while True:
            try:
                x1 = float(input(
                    'Please enter a value for the minimum x coordinate (in '
                    'metres - accepts scientific notation):') or -0.005)
            except ValueError:
                print('Invalid input. Please try again.')
                continue
            else:
                break
        while True:
            try:
                x2 = float(input(
                    'Please enter a value for the maximum x coordinate (in '
                    'metres - accepts scientific notation):') or 0.005)
            except ValueError:
                print('Invalid input. Please try again.')
                continue
            else:
                break
        while True:
            try:
                x1_prime = float(input(
                    'Please enter a value for the minimum horizontal aperture '
                    'limit (in metres - accepts scientific notation):')
                    or -1E-5)
            except ValueError:
                print('Invalid input. Please try again.')
                continue
            else:
                break
        while True:
            try:
                x2_prime = float(input(
                    'Please enter a value for the maximum horizontal aperture '
                    'limit (in metres - accepts scientific notation):')
                    or 1E-5)
            except ValueError:
                print('Invalid input. Please try again.')
                continue
            else:
                break
        while True:
            try:
                y1 = float(input(
                    'Please enter a value for the minimum y coordinate (in '
                    'metres - accepts scientific notation):') or -0.005)
            except ValueError:
                print('Invalid input. Please try again.')
                continue
            else:
                break
        while True:
            try:
                y2 = float(input(
                    'Please enter a value for the maximum y coordinate (in '
                    'metres - accepts scientific notation):') or 0.005)
            except ValueError:
                print('Invalid input. Please try again.')
                continue
            else:
                break
        while True:
            try:
                y1_prime = float(input(
                    'Please enter a value for the minimum vertical aperture '
                    'limit (in metres - accepts scientific notation):')
                    or -1E-5)
            except ValueError:
                print('Invalid input. Please try again.')
                continue
            else:
                break
        while True:
            try:
                y2_prime = float(input(
                    'Please enter a value for the maximum vertical aperture '
                    'limit (in metres - accepts scientific notation):')
                    or 1E-5)
            except ValueError:
                print('Invalid input. Please try again.')
                continue
            else:
                break
        while True:
            try:
                z = float(input('Please enter a value for the distance between'
                                ' the aperture and the screen (in metres - '
                                'accepts scientific notation): ') or 0.02)
            except ValueError:
                print('Invalid input. Please try again.')
                continue
            else:
                break

        k = 2*np.pi/_lambda
        x = np.linspace(x1, x2, N)
        y = np.linspace(y1, y2, N)
        x_prime = np.linspace(x1_prime, x2_prime, N)
        y_prime = np.linspace(y1_prime, y2_prime, N)
        E = np.zeros((N, N))

        E_x = simpson_integral(x_prime, k, x, z, N)
        E_y = simpson_integral(y_prime, k, y, z, N)

        # combine E_x and E_y into E
        for i in range(N):
            for j in range(N):
                E[i, j] = E_x[i] * E_y[j]

        plt.imshow(E, cmap=cm.YlOrRd_r)
        plt.title('2-Dimensional Intensity Map for the Diffraction Fringes on '
                  'the Screen', size=16)
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Relative Intensity')
        plt.xlabel('Relative Horizontal Screen Coordinate', size=12)
        plt.ylabel('Relative Vertical Screen Coordinate', size=12)
        plt.show()
