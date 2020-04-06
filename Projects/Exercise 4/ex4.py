#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author:  Francis Taylor
# Most comments are above the line they are commenting


# TODO
# fix text in CLI
# animate graphs
# add solar system values as options
# make gui?


# Reference
# Define a row vector
# v = array([[10, 20, 30]])

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim  # for animating the graphs oh yea yea

# define constants
G = 6.6743015E-11
M = 5.9722E24  # mass of earth


# def velocity_comp(x: float):
#     """
#     Evaluates the velocity component (f1 and f2)
#
#     v_x
#     """
#     # do we really need to evaluate this?
#     # do we really need y here?
#     # vis-viva eqn ?
#     r = np.sqrt(np.square(x) + np.square(y))


def gravitational_comp(x: float, y: float, _dir: str):
    """
    Evaluate f3 (dv_x/dt) and f4 (dv_y/dt).

    Parameters
    ----------
        x : float
            The x position.
        y : float
            The y position.
        _dir : str
            The direction to evaluate in.

    Returns
    -------
        acceleration : float
            The acceleration of the body at the current time.

    """
    if _dir == 'x':
        return -G*M*x/(np.square(x) + np.square(y))**(3/2)
    elif _dir == 'y':
        return -G*M*y/(np.square(x) + np.square(y))**(3/2)


def stepping_equations(x, y, v_x, v_y, t, delta_t, k1, k2, k3, k4, n):
    """
    Calculates x, y, v_x & v_y for each time-step.

    Parameters
    ----------
        x : array
            An array of values for the x-component of position
        y : array
            An array of values for the y-component of position
        v_x : array
            An array of values for the x-component of velocity
        v_y : array
            An array of values for the y-component of velocity
        delta_t : float
            The timestep to iterate over.
        n : int
            The number of data points to iterate over.
        k1 : array
            An array of values for k1
        k2 : array
            An array of values for k2
        k3 : array
            An array of values for k3
        k4 : array
            An array of values for k4

    Returns
    -------
        x : array
            An array of values for the x-component of position
        y : array
            An array of values for the y-component of position
        v_x : array
            An array of values for the x-component of velocity
        v_y : array
            An array of values for the y-component of velocity
        t : array
            An array of time values.
    """
    for i in range(0, n):
        x[i+1] = x[i] + delta_t/6 * \
            (k1[i, 0] + 2*k2[i, 0] + 2*k3[i, 0] + k4[i, 0])
        y[i+1] = y[i] + delta_t/6 * \
            (k1[i, 1] + 2*k2[i, 1] + 2*k3[i, 1] + k4[i, 1])
        v_x[i+1] = v_x[i] + delta_t/6 * \
            (k1[i, 2] + 2*k2[i, 2] + 2*k3[i, 2] + k4[i, 2])
        v_y[i+1] = v_y[i] + delta_t/6 * \
            (k1[i, 3] + 2*k2[i, 3] + 2*k3[i, 3] + k4[i, 3])
        t[i+1] = t[i] + delta_t
    return x, y, v_x, v_y, t


def define_k(x, y, v_x, v_y, delta_t, i):
    """
    Create and handle all the values required for k1 through k4.

    Reference: k1[x, y, vx, vy]

    Parameters
    ----------
        x : array
            Values for the x-component of position
        y : array
            Values for the y-component of position
        v_x : array
            Values for the x-component of velocity
        v_y : array
            Values for the y-component of velocity
        delta_t : float
            Timestep to iterate over.
        i : int
            Index of the sequence.

    Returns
    -------
        k1 : array
            Values for k1
        k2 : array
            Values for k2
        k3 : array
            Values for k3
        k4 : array
            Values for k4

    """

    k1[i, 0] = v_x
    k1[i, 1] = v_y
    k1[i, 2] = gravitational_comp(x, y, 'x')
    k1[i, 3] = gravitational_comp(x, y, 'y')

    k2[i, 0] = v_x + delta_t*k1[i, 2]/2
    k2[i, 1] = v_y + delta_t*k1[i, 3]/2
    k2[i, 2] = gravitational_comp(
        x + delta_t*k1[i, 0]/2, y + delta_t*k1[i, 1]/2, 'x')
    k2[i, 3] = gravitational_comp(
        x + delta_t*k1[i, 0]/2, y + delta_t*k1[i, 1]/2, 'y')

    k3[i, 0] = v_x + delta_t*k2[i, 2]/2
    k3[i, 1] = v_y + delta_t*k2[i, 3]/2
    k3[i, 2] = gravitational_comp(
        x + delta_t*k2[i, 0]/2, y + delta_t*k2[i, 1]/2, 'x')
    k3[i, 3] = gravitational_comp(
        x + delta_t*k2[i, 0]/2, y + delta_t*k2[i, 1]/2, 'y')

    k4[i, 0] = v_x + delta_t*k3[i, 2]/2
    k4[i, 1] = v_y + delta_t*k3[i, 3]/2
    k4[i, 2] = gravitational_comp(
        x + delta_t*k3[i, 0]/2, y + delta_t*k3[i, 1]/2, 'x')
    k4[i, 3] = gravitational_comp(
        x + delta_t*k3[i, 0]/2, y + delta_t*k3[i, 1]/2, 'y')

    return k1[i, 0], k1[i, 1], k1[i, 2], k1[i, 3], k2[i, 0], k2[i, 1], k2[i, 2], k2[i, 3], k3[i, 0], k3[i, 1], k3[i, 2], k3[i, 3], k4[i, 0], k4[i, 1], k4[i, 2], k4[i, 3]


def plot_graphs(x, y):
    """
    Handles plotting & animating the graphs.

    Parameters
    ----------
        x : array
            Position

    Returns
    -------
    """

    plt.plot(x, y)
    plt.xlabel('Relative distance in x direction (m)')
    plt.ylabel('Relative distance in y direction (m)')
    # creates a green dot to represent the Earth
    plt.scatter(0, 0, s=2E2, color='green')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


user_input = '0'
while user_input != 'q':
    user_input = input(
        '\nChoose an option:'
        '\na: circular orbit around the Earth,'
        '\nb: elliptical orbit around the Earth,'
        '\nc: rocket slingshot around the Moon,'
        '\n or q to quit.'
    ).lower()

    if user_input == 'a':
        print('4th order Runge-Kutta method')
        # loop until valid input
        # while True:
        try:
            # allows the user to change the paramters used in the
            # calculation
            custom_values = str(input(
                "Would you like to specify the values used in the problem?"
                " (y/n) ")).lower()
        except ValueError:
            # not a string
            print("Invalid input. Please try again.")
            continue
        if custom_values == 'y':
            # do stuff
            print("Using custom values.")
            # do stuff
            x[0] = float(input(
                'Enter a value for the start distance from the Earth (for best'
                ' results enter a number between 7,000,000 and 20,000,000): '))
            break
        elif custom_values == 'n':
            print("Using default values.")
            delta_t = 2
            n = 30000
            x = y = v_x = v_y = t = np.zeros(n)
            k1 = k2 = k3 = k4 = np.zeros(shape=(n, 4))

            y[0] = 0
            v_x[0] = 0
            v_y[0] = np.sqrt(G*M/x[0])
            print('defining k')

            # iterate here and call functions, rather than iterating in fns

            for i in range(0, n):
                k1[i, 0], k1[i, 1], k1[i, 2], k1[i, 3], k2[i, 0], k2[i, 1], k2[i, 2], k2[i, 3], k3[i, 0], k3[i,
                                                                                                             1], k3[i, 2], k3[i, 3], k4[i, 0], k4[i, 1], k4[i, 2], k4[i, 3] = define_k(x, y, v_x, v_y, delta_t, i)

            # k1, k2, k3, k4 = define_k(x, y, v_x, v_y, delta_t, n)
            print('time stepping')
            x, y, v_x, v_y, t = stepping_equations(
                x, y, v_x, v_y, t, delta_t, k1, k2, k3, k4, n)
            print('plotting')
            plot_graphs(x, y)
            print('plotted?')
            # break
