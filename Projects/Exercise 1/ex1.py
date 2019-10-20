#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# All comments are above the line they are commenting

# cexprtk is a cython wrapper around "C++ Mathematical Expression Toolkit
# Library (ExprTk)"
# installation: `pip install cexprtk`
# https://bitbucket.org/mjdr/cexprtk
try:
    import cexprtk
    from cexprtk._exceptions import ParseException
    parser_installed = True
except ImportError:
    parser_installed = False
    pass
import numpy as np
from tabulate import tabulate
import pylab


def arc_tan(x, N):
    '''
    Computes arctan(x) using a Taylor series expansion.

    Parameters
    ---------
    x : float
        The value for which the arctangent is being calculated.
    N : int
        The number of terms to iterate over.

    Returns
    -------
    sum : float
        The sum of the aray of values of arctan(x) computed using a Taylor series.
    '''
    # arctan(x) = sum of [(-1)^n]/[2n+1] * x^(2n+1)
    # N -> infinity, |x| <= 1

    arctan = np.zeros(N)
    try:
        if abs(x) <= 1:
            for i in range(0, N):
                arctan[i] = (((-1)**i)/(2*i+1)) * x**(2*i+1)
            sum = np.sum(arctan)

        # different mathematical function if |x| > 1
        elif abs(x) > 1:
            x_inverse = np.reciprocal(x)
            if x > 0:
                for i in range(0, N):
                    arctan[i] = (((-1)**i)/(2*i+1)) * (x_inverse)**(2*i+1)
                sum = np.pi/2 - np.sum(arctan)
            elif x < 0:
                for i in range(0, N):
                    arctan[i] = (((-1)**i)/(2*i+1)) * (x_inverse)**(2*i+1)
                sum = -1*(np.pi/2) - np.sum(arctan)

        return sum

    except Exception as e:
        print("An error was encountered. The following trace was generated: "
              "{}".format(e))


user_input = '0'
while user_input != 'q':
    user_input = input(
        '\nChoose option "a" or "b", or type "q" to quit: ')

    if user_input == 'a':
        print('You have chosen part a.')

        try:
            x_input = input('Enter a value for x: ')
            if parser_installed:
                # parses mathematical expressions in the input field
                x = cexprtk.evaluate_expression(x_input, {})
            else:
                x = float(x_input)

        except ValueError:
            print("Invalid input. \"{}\" is not a valid "
                  "value.".format(x_input))

        except ParseException:
            print("Invalid syntax: \"{}\" Please make sure your function is "
                  "correctly formatted using brackets. e.g. 'sqrt(x)' "
                  "rather than 'sqrtx'".format(x_input))

        try:
            # allows the user to input the number of terms to iterate over
            N = int(input('Enter a value for N: '))

            # stops the user selecting a too large value for N to improve
            # performance
            if N > 1000000:
                print("Value of N entered is too large. Using N = 1,000,000.")
                N = 1000000

        except ValueError:
            print("Invalid input for N. Using N = 10,000.")
            N = 10000

        print('The arctan of {0} using the Taylor expansion over {1} terms is:'
              ' {2:.5f}'.format(
                  x, N, arc_tan(float(x), N)))

    elif user_input == 'b':
        print('You have chosen part b.')

        try:
            # allows the user to input the number of terms to iterate over
            N = int(input('Enter a value for N: '))

            # stops the user selecting a too large value for N to improve
            # performance
            if N > 1000000:
                print("Value of N entered is too large. Using N = 1,000,000.")
                N = 1000000

        except ValueError:
            print("Invalid input for N. Using N = 10,000.")
            N = 10000

        # set x to lower bound specified in brief
        x = -2.0
        # initialise arrays for 40 steps of 0.1, to allow for -2.0 <= x <= 2.0
        arctan = np.zeros(41)
        np_arctan = np.zeros(41)
        x_array = np.zeros(41)
        delta = np.zeros(41)

        # this loop iterates 40 times and adds the value of x, arctan(x), and
        # the value computed by numpy to an array
        for i in range(0, 41):
            x_array[i] = x
            arctan[i] = arc_tan(x, N)
            np_arctan[i] = np.arctan(x)
            delta[i] = abs(arctan[i] - np_arctan[i])
            x += 0.1

        # transforms the arrays into 4x41 array for the tabulate package
        array = np.column_stack((x_array, arctan, np_arctan, delta))
        print(tabulate(array, headers=[
              "x", "arctan", "NumPy arctan", "delta"], tablefmt="psql",
            numalign="right", floatfmt=(".5f", ".5f", ".5f", ".3e")))

        choice = input('Would you like to generate a graph? (y/n): ')
        if choice == 'y':
            # settings for the pylab plot
            pylab.plot(x_array, arctan, linewidth=1,
                       label='Taylor Series Arctan')
            pylab.plot(x_array, np_arctan, linewidth=1, label='Numpy Arctan')
            pylab.xlabel("x")
            pylab.ylabel("arctan(x)")
            pylab.legend()
            pylab.show()
        elif choice == 'n':
            pass
        else:
            print('Not a valid choice.')

    elif user_input != 'q':
        print('\'{}\' is not a valid choice.'.format(user_input))

print('You have chosen to finish - exiting.')
