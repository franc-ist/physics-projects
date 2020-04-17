#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author:  Francis Taylor
# Most comments are above the line they are commenting

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim  # for animating the graphs
from tqdm import tqdm

# define constants
G = 6.6743015E-11
M_E = 5.9722E24  # mass of Earth
M_M = 7.342E22  # mass of the Moon
m = 1.5E6  # roughly the mass of Falcon Heavy
r_E = 6.3781E6  # radius of Earth
r_M = 1.7374E6  # radius of Moon
r = 384.403E6  # distance between Earth & Moon


def gravitational_comp(x: float, y: float, _dir: str, fn: str):
    """
    Evaluate f3 (dv_x/dt) and f4 (dv_y/dt).

    Parameters
    ----------
        x : float
            x-component of position.
        y : float
            y-component of position.
        _dir : str
            Direction to evaluate in.
        fn : str
            Specifies which equation of motion to use.

    Returns
    -------
        acceleration : float
            Acceleration of the body at the current time.

    """
    if fn == 'earth':
        if _dir == 'x':
            return -G*M_E*x/((np.square(x) + np.square(y))**(3/2))
        elif _dir == 'y':
            return -G*M_E*y/((np.square(x) + np.square(y))**(3/2))

    elif fn == 'moon':
        if _dir == 'x':
            return (-G*M_E*x/(np.square(x) + np.square(y))**(3/2)) - (G*M_M*x/((np.square(x) + np.square(y-r))**(3/2)))
        elif _dir == 'y':
            return (-G*M_E*y/(np.square(x) + np.square(y))**(3/2)) - (G*M_M*(y-r)/((np.square(x) + np.square(y-r))**(3/2)))


def calculate_energies(x: float, y: float, v_x: float, v_y: float):
    """
    Calculate gravitational and kinetic components of the energy of the rocket.

    Parameters
    ----------
        x : float
            x-component of position.
        y : float
            y-comnponent of position.
        v_x : float
            Tangential component of velocity.
        v_y : float
            Radial component of velocity.

    Returns
    -------
        E_k : float
            Kinetic component of total energy.
        E_g : float
            Gravitation component of total energy.
        E_t : float
            Total energy of the rocket.

    """


def stepping_equations(x, y, v_x, v_y, t, delta_t: float, k1, k2, k3, k4):
    """
    Calculate x, y, v_x & v_y for each time-step.

    Parameters
    ----------
        x: array_like
            An array of values for the x-component of position
        y: array_like
            An array of values for the y-component of position
        v_x: array_like
            An array of values for the x-component of velocity
        v_y: array_like
            An array of values for the y-component of velocity
        delta_t: float
            The timestep to iterate over.
        n: int
            The number of data points to iterate over.
        k1: array_like
            An array of values for k1
        k2: array_like
            An array of values for k2
        k3: array_like
            An array of values for k3
        k4: array_like
            An array of values for k4

    Returns
    -------
        x: array_like
            An array of values for the x-component of position
        y: array_like
            An array of values for the y-component of position
        v_x: array_like
            An array of values for the x-component of velocity
        v_y: array_like
            An array of values for the y-component of velocity
        t: array_like
            An array of time values.

    """
    x += delta_t/6 * \
        (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
    y += delta_t/6 * \
        (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    v_x += delta_t/6 * \
        (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
    v_y += delta_t/6 * \
        (k1[3] + 2*k2[3] + 2*k3[3] + k4[3])
    t += delta_t
    return x, y, v_x, v_y, t


def define_k(x, y, v_x, v_y, delta_t: float, orbit: str = None):
    """
    Create and handle all the values required for k1 through k4.

    Reference: k1[x, y, vx, vy]

    Parameters
    ----------
        x: array_like
            Values for the x-component of position.
        y: array_like
            Values for the y-component of position.
        v_x: array_like
            Values for the x-component of velocity.
        v_y: array_like
            Values for the y-component of velocity.
        delta_t: float
            Timestep to iterate over.
        orbit : str
            Change to using formula for moon slingshot.

    Returns
    -------
        k1: array_like
            Values for k1
        k2: array_like
            Values for k2
        k3: array_like
            Values for k3
        k4: array_like
            Values for k4

    """
    if orbit == 'moon':
        k1x = v_x
        k1y = v_y
        k1vx = gravitational_comp(x, y, 'x', 'moon')
        k1vy = gravitational_comp(x, y, 'y', 'moon')

        k2x = v_x + delta_t*k1vx/2
        k2y = v_y + delta_t*k1vy/2
        k2vx = gravitational_comp(
            x + delta_t*k1x/2, y + delta_t*k1y/2, 'x', 'moon')
        k2vy = gravitational_comp(
            x + delta_t*k1x/2, y + delta_t*k1y/2, 'y', 'moon')

        k3x = v_x + delta_t*k2vx/2
        k3y = v_y + delta_t*k2vy/2
        k3vx = gravitational_comp(
            x + delta_t*k2x/2, y + delta_t*k2y/2, 'x', 'moon')
        k3vy = gravitational_comp(
            x + delta_t*k2x/2, y + delta_t*k2y/2, 'y', 'moon')

        k4x = v_x + delta_t*k3vx/2
        k4y = v_y + delta_t*k3vy/2
        k4vx = gravitational_comp(
            x + delta_t*k3x/2, y + delta_t*k3y/2, 'x', 'moon')
        k4vy = gravitational_comp(
            x + delta_t*k3x/2, y + delta_t*k3y/2, 'y', 'moon')
    else:
        k1x = v_x
        k1y = v_y
        k1vx = gravitational_comp(x, y, 'x', 'earth')
        k1vy = gravitational_comp(x, y, 'y', 'earth')

        k2x = v_x + delta_t*k1vx/2
        k2y = v_y + delta_t*k1vy/2
        k2vx = gravitational_comp(
            x + delta_t*k1x/2, y + delta_t*k1y/2, 'x', 'earth')
        k2vy = gravitational_comp(
            x + delta_t*k1x/2, y + delta_t*k1y/2, 'y', 'earth')

        k3x = v_x + delta_t*k2vx/2
        k3y = v_y + delta_t*k2vy/2
        k3vx = gravitational_comp(
            x + delta_t*k2x/2, y + delta_t*k2y/2, 'x', 'earth')
        k3vy = gravitational_comp(
            x + delta_t*k2x/2, y + delta_t*k2y/2, 'y', 'earth')

        k4x = v_x + delta_t*k3vx/2
        k4y = v_y + delta_t*k3vy/2
        k4vx = gravitational_comp(
            x + delta_t*k3x/2, y + delta_t*k3y/2, 'x', 'earth')
        k4vy = gravitational_comp(
            x + delta_t*k3x/2, y + delta_t*k3y/2, 'y', 'earth')

    return k1x, k1y, k1vx, k1vy, k2x, k2y, k2vx, k2vy, k3x, k3y, k3vx, k3vy, k4x, k4y, k4vx, k4vy


def _custom_values():
    """
    Allow the user to specify custom values for the orbits.

    Returns
    -------
        x: float
            Initial horizontal component of position.
        y: float
            Initial vertical component of position.
        v_x: float
            Initial horizontal component velocity.
        v_y: float
            Initial vertical component velocity.
        T: float
            Time period to simulate an orbit for.
        delta_t: float
            The time period step(dt). Defaults to 2.0 s.

    """
    print('Please enter the values you wish to use. Leave blank for '
          'the default. (Scientific notation is accepted e.g. 10E6)')

    # loop each input until valid value
    while True:
        try:
            T = float(input(
                "Please enter a value for T (time period to simulate over) in "
                "seconds: ") or 60000)
            if (T <= 0):
                raise ValueError(
                    'T must be larger than 0. Please try again.')
        except ValueError:
            print('Invalid input.')
            continue
        else:
            break

    while True:
        try:
            delta_t = float(
                input("Please input a value for \u0394t in seconds: ") or 2)
            if (delta_t <= 0):
                raise ValueError(
                    '\u0394t must be larger than 0. Please try again.')
        except ValueError:
            print('Invalid input. Please try again.')
            continue
        else:
            break

    n = round(T/delta_t)

    while True:
        try:
            x = float(
                input("Please enter a value for the initial tangential "
                      "position: ") or 7E6)
            y = float(
                input("Please enter a value for the initial radial "
                      "position: ") or 0)
            if (np.sqrt(np.square(x) + np.square(y)) <= r_E):
                raise ValueError('Oops! That position is smaller than the '
                                 'radius of the Earth. Please enter a larger '
                                 'value for x or y.')
        except ValueError:
            print('Invalid input. Please try again.')
            continue
        else:
            break

    while True:
        try:
            v_x = float(
                input("Please enter a value for the initial tangential "
                      "speed: ") or 7500)
        except ValueError:
            print('Invalid input. Please try again.')
            continue
        else:
            break

    while True:
        try:
            v_y = float(
                input("Please enter a value for the initial radial "
                      "speed: ") or 0)
        except ValueError:
            print('Invalid input. Please try again.')
            continue
        else:
            break

    # create arrays for positional variables
    ax = np.zeros(n)
    ay = np.zeros(n)
    av_x = np.zeros(n)
    av_y = np.zeros(n)

    # assign custom values to 0th index
    ax[0] = x
    ay[0] = y
    av_x[0] = v_x
    av_y[0] = v_y

    return ax, ay, av_x, av_y, delta_t, n, T


def plot_graphs(x, y):
    """
    Handle plotting & animating the graphs.

    Parameters
    ----------
        x: array_like
            Position values for the x-component of motion
        y: array_like
            Position values for the y-component of motion

    """
    plt.plot(x, y)
    plt.xlabel('Relative distance in x direction (m)')
    plt.ylabel('Relative distance in y direction (m)')
    # creates a green dot to represent the Earth
    plt.scatter(0, 0, s=6.37E2, color='green')
    if user_input == 'c':
        plt.scatter(0, r, s=1.74E2, color='grey')
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
        print('Simulating a circular orbit around the Earth using a 4th order '
              'Runge-Kutta method')
        # loop until valid input
        while True:
            try:
                # allows the user to change the paramters used in the
                # calculation
                custom_values = str(input(
                    "Would you like to specify the values used in the simulation?"
                    " (y/n) ")).lower()
                if custom_values == 'y' or custom_values == 'n':
                    break
                else:
                    raise ValueError()
            except ValueError:
                # not a string
                print("Invalid input. Please try again.")
                continue
        if custom_values == 'y':
            x, y, v_x, v_y, delta_t, n, T = _custom_values()
            t = np.zeros(n)
            print('Using custom values.\nT = {}s\n\u0394t = {}s\nx = {}m\ny = {}m\nv_x = {:.2f}m/s\nv_y = {:.2f}m/s'.format(
                T, delta_t, x[0], y[0], v_x[0], v_y[0]))
        elif custom_values == 'n':
            delta_t = 2
            n = 30000
            # create arrays for positional variables
            x = np.zeros(n)
            y = np.zeros(n)
            v_x = np.zeros(n)
            v_y = np.zeros(n)
            t = np.zeros(n)

            x[0] = 10000000
            y[0] = 0
            v_x[0] = 0
            v_y[0] = np.sqrt(G*M_E/x[0])
            print(
                'Using default values.\nT = 60000s\n\u0394t = 2s\nx = 10E6m\nv = {:.2f}m/s'.format(v_y[0]))

        # create arrays for rk4
        k1 = np.zeros(shape=(n, 4))
        k2 = np.zeros(shape=(n, 4))
        k3 = np.zeros(shape=(n, 4))
        k4 = np.zeros(shape=(n, 4))

        # iterates through the rk4 values and creates a progress bar
        for i in tqdm(range(0, n-1), desc='Simulating...', mininterval=0.25, bar_format='{desc} {percentage:3.0f}%|{bar}| Iteration {n_fmt} of {total_fmt} [Estimated time remaining: {remaining}, {rate_fmt}]'):
            k1[i, 0], k1[i, 1], k1[i, 2], k1[i, 3], k2[i, 0], k2[i, 1], k2[i, 2], k2[i, 3], k3[i, 0], k3[i, 1], k3[i,
                                                                                                                   2], k3[i, 3], k4[i, 0], k4[i, 1], k4[i, 2], k4[i, 3] = define_k(x[i], y[i], v_x[i], v_y[i], delta_t)
            # i have no idea why the linter is breaking the line in two in such a weird way?

            x[i+1], y[i+1], v_x[i+1], v_y[i+1], t[i+1] = stepping_equations(
                x[i], y[i], v_x[i], v_y[i], t[i], delta_t, k1[i, ], k2[i, ], k3[i, ], k4[i, ])

            # break fn if we hit the Earth
            if (np.sqrt(np.square(x[i] + np.square(y[i])) <= r_E)):
                break

        plot_graphs(x, y)  # anim?

    elif user_input == 'b':
        print('Simulating an elliptical orbit around the Earth.')
        # loop until valid input
        while True:
            try:
                # allows the user to change the paramters used in the
                # calculation
                custom_values = str(input(
                    "Would you like to specify the values used in the simulation?"
                    " (y/n) ")).lower()
                if custom_values == 'y' or custom_values == 'n':
                    break
                else:
                    raise ValueError()
            except ValueError:
                # not a string
                print("Invalid input. Please try again.")
                continue
        if custom_values == 'y':
            x, y, v_x, v_y, delta_t, n, T = _custom_values()
            t = np.zeros(n)
            print('Using custom values.\nT={}s\n\u0394t={}s\nx={}m\ny={}m\nv_x={: .2f}m/s\nv_y={: .2f}m/s'.format(
                T, delta_t, x[0], y[0], v_x[0], v_y[0]))

        elif custom_values == 'n':
            delta_t = 2
            n = 30000
            # create arrays for positional variables
            x = np.zeros(n)
            y = np.zeros(n)
            v_x = np.zeros(n)
            v_y = np.zeros(n)
            t = np.zeros(n)

            x[0] = 10000000
            y[0] = 0
            v_x[0] = 0
            v_y[0] = 7500
            print(
                'Using default values.\nT = 60000s\n\u0394t = 2s\nx = 10E6m\nv = {:.2f}m/s'.format(v_y[0]))

        # create arrays for rk4
        k1 = np.zeros(shape=(n, 4))
        k2 = np.zeros(shape=(n, 4))
        k3 = np.zeros(shape=(n, 4))
        k4 = np.zeros(shape=(n, 4))

        # iterates through the rk4 values and creates a progress bar
        for i in tqdm(range(0, n-1), desc='Simulating...', mininterval=0.25, bar_format='{desc} {percentage:3.0f}%|{bar}| Iteration {n_fmt} of {total_fmt} [Estimated time remaining: {remaining}, {rate_fmt}]'):
            k1[i, 0], k1[i, 1], k1[i, 2], k1[i, 3], k2[i, 0], k2[i, 1], k2[i, 2], k2[i, 3], k3[i, 0], k3[i, 1], k3[i,
                                                                                                                   2], k3[i, 3], k4[i, 0], k4[i, 1], k4[i, 2], k4[i, 3] = define_k(x[i], y[i], v_x[i], v_y[i], delta_t)
            # i have no idea why the linter is breaking the line in two in such a weird way?

            x[i+1], y[i+1], v_x[i+1], v_y[i+1], t[i+1] = stepping_equations(
                x[i], y[i], v_x[i], v_y[i], t[i], delta_t, k1[i, ], k2[i, ], k3[i, ], k4[i, ])

            # break fn if we hit the Earth
            if (np.sqrt(np.square(x[i]) + np.square(y[i])) <= r_E):
                break

        plot_graphs(x, y)  # anim?

    elif user_input == 'c':
        print('Simulating a slingshot orbit around the moon.')
        # loop until valid input
        while True:
            try:
                # allows the user to change the paramters used in the
                # calculation
                custom_values = str(input(
                    "Would you like to specify the values used in the simulation?"
                    " (y/n) ")).lower()
                if custom_values == 'y' or custom_values == 'n':
                    break
                else:
                    raise ValueError()
            except ValueError:
                # not a string
                print("Invalid input. Please try again.")
                continue
        if custom_values == 'y':
            x, y, v_x, v_y, delta_t, n, T = _custom_values()
            t = np.zeros(n)
            print('Using custom values.\nT = {}s\n\u0394t = {}s\nx = {}m\ny = {}m\nv_x = {:.2f}m/s\nv_y = {:.2f}m/s'.format(
                T, delta_t, x[0], y[0], v_x[0], v_y[0]))

        elif custom_values == 'n':
            delta_t = 2
            n = 500000
            # create arrays for positional variables
            x = np.zeros(n)
            y = np.zeros(n)
            v_x = np.zeros(n)
            v_y = np.zeros(n)
            t = np.zeros(n)

            x[0] = 0
            y[0] = -7E6
            v_x[0] = 10591
            v_y[0] = 0
            print(
                'Using default values.\nT = 1000000s\n\u0394t = 2s\nInitial Orbit Height = {:.0f}m\nv = {:.2f}m/s'.format(y[0], v_x[0]))

        # create arrays for rk4
        k1 = np.zeros(shape=(n, 4))
        k2 = np.zeros(shape=(n, 4))
        k3 = np.zeros(shape=(n, 4))
        k4 = np.zeros(shape=(n, 4))

        # iterates through the rk4 values and creates a progress bar
        for i in tqdm(range(0, n-1), desc='Simulating...', mininterval=0.25, bar_format='{desc} {percentage:3.0f}%|{bar}| Iteration {n_fmt} of {total_fmt} [Estimated time remaining: {remaining}, {rate_fmt}]'):
            k1[i, 0], k1[i, 1], k1[i, 2], k1[i, 3], k2[i, 0], k2[i, 1], k2[i, 2], k2[i, 3], k3[i, 0], k3[i, 1], k3[i,
                                                                                                                   2], k3[i, 3], k4[i, 0], k4[i, 1], k4[i, 2], k4[i, 3] = define_k(x[i], y[i], v_x[i], v_y[i], delta_t, 'moon')
            # i have no idea why the linter is breaking the line in two in such a weird way?

            x[i+1], y[i+1], v_x[i+1], v_y[i+1], t[i+1] = stepping_equations(
                x[i], y[i], v_x[i], v_y[i], t[i], delta_t, k1[i, ], k2[i, ], k3[i, ], k4[i, ])

            # break fn if we hit the Earth or Moon
            if (np.sqrt(np.square(x[i]) + np.square(y[i])) <= r_E) and y[i] < r/2:
                print("\nCrashed into the Earth!")
                break
            elif (abs(np.sqrt(np.square(x[i]) + np.square(y[i] - r))) <= (r_M)) and y[i] > r/2:
                print("\nCrashed into the Moon!")
                break

        plot_graphs(x, y)  # anim?
