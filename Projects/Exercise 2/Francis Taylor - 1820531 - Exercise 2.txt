#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author:  Francis Taylor
# Most comments are above the line they are commenting

import numpy as np
import matplotlib.pyplot as plt


# define constants
g = -9.80665
h = 7640
v_sound = 343.2  # in air at sea level


def speed_of_sound(height: float):
    '''
    Calculates the speed of sound at various altitudes, to allow for a
    horizontal line to be added to the graphs showing the speed of sound.

    https://www.grc.nasa.gov/WWW/BGH/atmosmet.html
    http://hyperphysics.phy-astr.gsu.edu/hbase/Sound/souspe3.html

    Parameters
    ----------
        height : float
            The height at which to calculate the speed of sound.


    Returns
    -------
        v_sound : float
            The speed of sound at the given altitude.
    '''

    temp = 273
    # temperature in the atmosphere depends on the layer of the atmosphere
    if height > 25000:
        temp += -131.21 + 0.00299*height
    elif 11000 < h <= 25000:
        temp += -56.46
    else:
        temp += 15.04 - 0.00649*height
    # speed of sound is dependent on the temperature of the medium it is
    # travelling through
    v_sound = np.sqrt((1.4*8.314*temp)/0.02895)
    return v_sound


def _custom_values(fn: str):
    '''
    Allows the user to specify custom values for the analytical predictions,
    Eulers method, and varying air density problems.

    Parameters
    ----------
        fn : str
            The function that the values will be used for.


    Returns
    -------
        n : int
            The number of terms to iterate over. Defaults to 1000.
        y0 : float
            Initial position of the object. Defaults to 1000 m.
        v0 : float
            Initial velocity of the object. Defaults to 0 m/s.
        mass : float
            The mass of the object in freefall. Defaults to 100 kg.
        x_section_area : float
            The cross sectional area of the object. Defaults to 0.95m^2.
        drag_coefficient : float
            The drag coefficient of the object. Defaults to 1.0.
        air_density : float
            The density of the air. Defaults to 1.2 kg/m^3.
        t0 : float[Optional]
            Initial time. Defaults to 0.0 s.
        delta_t : float[Optional]
            The time period step (dt). Defaults to 1.0 s.
        t_max : float[Optional]
            The final time value for the prediction. Defaults to 60.0s.
        t_min : float[Optional]
            The initial time value for the prediction range. Defaults to 0.0s.
    '''
    print('Please enter the values you wish to use. Leave blank for '
          'the default.')
    while True:
        try:
            n = abs(int(input("Please enter a value for n: ") or 1000))
            y0 = float(input("Please enter a value for y0: ") or 1000.00)
            v0 = float(input("Please enter a value for v0: ") or 0.0)
            mass = float(input(
                "Please enter a value for the mass of the object: ") or 100.0)
            x_section_area = float(input(
                "Please enter a value for the cross-sectional area of the "
                "object: ") or 0.95)
            drag_coefficient = float(input(
                "Please enter a value for the drag coefficient: ") or 1.0)
            air_density = float(input(
                "Please enter a value for the density of the air: ") or 1.2)

            # euler specific values
            if fn == 'euler':
                t0 = float(input("Please enter a value for t0: ") or 0.0)
                delta_t = float(input("Please enter a value for dt: ") or 1.0)
                return n, y0, v0, mass, x_section_area, drag_coefficient, air_density, t0, delta_t

            # analytical specific values
            elif fn == 'analytical':
                t_min = float(
                    input("Please enter a value for the lower bound of t (t_min): ") or 0.0)
                t_max = float(
                    input("Please enter a value for the upper bound of t (t_max): ") or 300.0)
                return n, y0, v0, mass, x_section_area, drag_coefficient, air_density, t_max, t_min
        except ValueError:
            print('Invalid input.')
            continue
        else:
            break


def plot_graphs(t, dep_var, dep_type, comparison: bool = False, comparison_t=None, comparison_dep_var=None, fn: str = 'Analytical'):
    '''
    Utitlity function to handle plotting of graphs.

    Parameters
    ----------
        t : array
            Array of time values to be plotted as the independent variable.
        dep_var : array
            Array of values to be plotted as the dependent variable.
        dep_type : str
            Specifies whether the dependent variable is height or velocity.
        comparison : bool[Optional]
            Determines whether to plot the analytical solution on the same
            graph as the Euler solution
        comparison_t : array[Optional]
            Array of time values to be plotted as the second independent
            variable.
        comparison_dep_var : array[Optional]
            Array of values to be plotted as the second dependent variable.
        fn : str[Optional]
            Determines which function should be labelled in the comparison
            graph.
    '''
    if comparison is True:
        plt.plot(t, dep_var, color='red', label='Euler')
        plt.plot(comparison_t, comparison_dep_var,
                 color='blue', label=fn)
        plt.legend()
    else:
        # no comparison
        plt.plot(t, dep_var)
        plt.xlabel('$Time (s)$', size=12)
    # velocity graph
    if dep_type == 'v':

        plt.title('Velocity-Time', size=22)
        plt.ylabel('$Velocity (m/s)$', size=12)
    # height graph
    elif dep_type == 'y':
        plt.title('Height-Time', size=22)
        plt.ylabel('$Height (m)$', size=12)
    plt.grid(alpha=0.7, linewidth=1)


def trim_zeros(v, y, t):
    '''
    Trims '0' values from the ends of the velocity and height arrays. Resizes
    the time array to the dimensions of the trimmed velocity and height arrays.

    Parameters
    ----------
        v : NumPy array
            Array of vertical velocity values to be trimmed.
        y : NumPy array
            Array of height values to be trimmed.
        t : NumPy array
            Array of time values to be resized to the dimensions of the
            trimmed velocity and height arrays


    Returns
    -------
        v_trimmed : NumPy array
            Array of vertical velocity values, with end zeros trimmed.
        y_trimmed : NumPy array
            Array of height values, with end zeros trimmed.
        t_v : NumPy array
            Time array resized to the dimensions of the v_trimmed array.
        t_y : NumPy array
            Time array resized to the dimensions of the y_trimmed array.
    '''

    # trims 0 values at the end of the array
    v_trimmed = np.trim_zeros(v, 'b')
    y_trimmed = np.trim_zeros(y, 'b')
    # trims time array to length of the velocity array
    m_v = len(v_trimmed)
    t_v = np.resize(t, m_v)
    # trims height array to length of the velocity array
    m_y = len(y_trimmed)
    t_y = np.resize(t, m_y)

    return v_trimmed, y_trimmed, t_v, t_y


def analytical_predictions(n: int = 200, y0: float = 1000.0, v0: float = 0.0, t_max: float = 300.0, t_min: float = 0.0, mass: float = 100.0, x_section_area: float = 0.95, drag_coefficient: float = 1.0, air_density: float = 1.2):
    '''
    Predicts the height and vertical speed of an object in freefall with
    constant gravitational acceleration and drag, using an analytical method.

    Parameters
    ----------
        n : int
            The number of terms to iterate over. Defaults to 1000.
        y0 : float
            Initial position of the object. Defaults to 1000 m.
        v0 : float
            Initial velocity of the object. Defaults to 0 m/s.
        t_max : float
            The final time value for the prediction. Defaults to 60.0s.
        t_min : float
            The initial time value for the prediction range. Defaults to 0.0s.
        mass : float
            The mass of the object in freefall. Defaults to 100 kg.
        x_section_area : float
            The cross sectional area of the object. Defaults to 0.95m^2.
        drag_coefficient : float
            The drag coefficient of the object. Defaults to 1.0.
        air_density : float
            The density of the air. Defaults to 1.2 kg/m^3.


    Returns
    -------
        v_vals : NumPy array
            An array of vertical speed values as time progresses for the
            object.
        y_vals : NumPy array
            An array of height values as time progresses for the object.
        t_v : NumPy array
            Time array resized to the dimensions of the v array.
        t_y : NumPy array
            Time array resized to the dimensions of the y array.
    '''
    k = (drag_coefficient*air_density*x_section_area)/2
    # initialise arrays
    t_vals = np.linspace(t_min, t_max, n)
    y_vals = np.zeros(n)
    v_vals = np.zeros(n)

    # set t=0 values
    y_vals[0], v_vals[0] = y0, v0

    # iteratively calculate y and v values for varying t
    for i in range(0, n):
        y_vals[i] = y0 - \
            ((mass/k) * np.log(np.cosh(np.sqrt((k*abs(g))/mass) * t_vals[i])))

        v_vals[i] = -1 * \
            np.sqrt((mass*abs(g))/k) * \
            np.tanh((np.sqrt((k*abs(g))/mass) * t_vals[i]))
        # stop if we hit the ground
        if y_vals[i] <= 0:
            break

    # trims trailing zeros and resizes time array
    return trim_zeros(v_vals, y_vals, t_vals)


def euler(n: int = 1000, y0: float = 1000.0, t0: float = 0.0, delta_t: float = 1.0, v0: float = 0.0, mass: float = 100.0, x_section_area: float = 0.95, drag_coefficient: float = 1.0, air_density: float = 1.2, var_air_dens: bool = False):
    '''
    Uses the Euler method to solve a second order ODE for an object in
    freefall with constant air density and gravitational acceleration.

    Parameters
    ----------
        n : int
            The number of terms to iterate over. Defaults to 1000.
        y0 : float
            Initial position of the object. Defaults to 1000 m.
        t0 : float
            Initial time. Defaults to 0.0 s.
        delta_t : float
            The time period step (dt). Defaults to 1.0 s.
        v0 : float
            Initial velocity of the object. Defaults to 0 m/s.
        mass : float
            The mass of the object in freefall. Defaults to 100 kg.
        x_section_area : float
            The cross sectional area of the object. Defaults to 0.95 m^2.
        drag_coefficient : float
            The drag coefficient of the object. Defaults to 1.0.
        air_density : float
            The density of the air. Defaults to 1.2 kg/m^3.
        var_air_dens : bool
            If true, uses the equation for varying air density. Defaults to
            false.


    Returns
    -------
        v_vals : NumPy array
            An array of vertical speed values as time progresses for the
            object.
        y_vals : NumPy array
            An array of height values as time progresses for the object.
        t_v : NumPy array
            Time array resized to the dimensions of the v array.
        t_y : NumPy array
            Time array resized to the dimensions of the y array.
    '''
    # initialise arrays
    v = np.zeros(n)
    t = np.zeros(n)
    y = np.zeros(n)
    # set t0 values of v, y and t
    v[0], t[0], y[0] = v0, t0, y0

    # redefine k using new, varying air density
    if var_air_dens is True:
        # initialise air density and k arrays for varying values
        ad = np.zeros(n)
        k = np.zeros(n)
        for i in range(0, n-1):
            ad[i] = air_density*np.exp(-1*y[i]/h)
            k[i] = (drag_coefficient*ad[i]*x_section_area)/2
            t[i+1] = t[i] + delta_t
            v[i+1] = v[i] - (delta_t * (-g + (k[i]/mass) * abs(v[i])*v[i]))
            y[i+1] = y[i] + (delta_t * v[i])

            # stop when we hit the ground
            if y[i+1] <= 0:
                break
    else:
        for i in range(0, n-1):
            k = (drag_coefficient*air_density*x_section_area)/2
            t[i+1] = t[i] + delta_t
            v[i+1] = v[i] - (delta_t * (-g + (k/mass) * abs(v[i])*v[i]))
            y[i+1] = y[i] + (delta_t * v[i])

            # stop when we hit the ground
            if y[i+1] <= 0:
                break

    # trims trailing zeros and resizes time array
    return trim_zeros(v, y, t)


user_input = '0'
while user_input != 'q':
    user_input = input(
        '\nChoose an option:'
        '\na: Plot graphs of vertical velocity and height as a function of '
        'time using an analytical method for a object in freefall,'
        '\nb: Plot graphs of vertical velocity and height as a function of '
        'time using Euler\'s method for an object in freefall,'
        '\nc: Same as option b, but with varying air density,'
        '\nOr type "q" to quit. ').lower()

    if user_input == 'a':
        print("Plotting the analytical predictions for height and vertical "
              "speed.")
        # loop until valid input
        while True:
            try:
                # allows the user to change the paramters used in the calculation
                custom_values = str(input(
                    "Would you like to specify the values used in the problem? (y/n) ")).lower()
            except ValueError:
                # not a string
                print("Invalid input. Please try again.")
                continue
            if custom_values == 'y':
                fn = 'analytical'
                # calls _custom_values() for the analytical solution to allow # the parameters used to be changed
                n, y0, v0, mass, x_section_area, drag_coefficient, air_density, t_max, t_min = _custom_values(
                    fn)
                # runs the analytical method using the custom values
                v, y, t_v, t_y = analytical_predictions(
                    n, y0, v0, t_max, t_min, mass, x_section_area, drag_coefficient, air_density)
                break
            elif custom_values == 'n':
                print("Using default values.")
                v, y, t_v, t_y = analytical_predictions()
                break
            else:
                print("Invalid input. Please try again.")
                continue

        # checks if v >= v_sound at any point in the freefall (as it is
        # variable), and plots v_sound on the graph
        v_sound = np.zeros(len(y))
        v_sound_passed = 343
        for i in range(0, len(y)):
            v_sound[i] = speed_of_sound(y[i])
            if abs(v[i]) >= abs(v_sound[i]):
                v_sound_passed = -1*v_sound[i]
                plt.axhline(v_sound_passed, color='orange', linestyle='--',
                            label='Speed of sound = {:.2f}m/s\n(at {:.1f}m) '.format(v_sound_passed, y[i]))
                plt.legend()
                break

        # plots graphs of velocity and height against time
        plot_graphs(t_v, v, 'v')
        plt.show()
        plot_graphs(t_y, y, 'y')
        plt.show()

    elif user_input == 'b':
        print("Solving the freefall of a body using Euler's method.")
        # loop until valid input
        while True:
            try:
                # allows the user to change the paramters used in the calculation
                custom_values = str(input(
                    "Would you like to specify the values used in the problem? (y/n) ")).lower()
            except ValueError:
                # not a string
                print("Invalid input. Please try again.")
                continue
            if custom_values == 'y':
                fn = 'euler'
                # calls _custom_values() for the euler solution to allow the
                # parameters used to be changed
                n, y0, v0, mass, x_section_area, drag_coefficient, air_density, t0, delta_t = _custom_values(
                    fn)
                # runs the euler method using the custom values
                v, y, t_v, t_y = euler(
                    n, y0, t0, delta_t, v0, mass, x_section_area, drag_coefficient, air_density)
                break
            elif custom_values == 'n':
                # uses default values
                print('Using default values.')
                v, y, t_v, t_y = euler()
                break
            else:
                print("Invalid input. Please try again.")
                continue
        while True:
            try:
                # allows the user to compare the euler result with the
                # analytical soln
                compare_graphs = str(input(
                    "Would you like to compare the results with the analytical solution? (y/n) ")).lower()
            except ValueError:
                # not a string
                print("Invalid input. Please try again.")
                continue
            if compare_graphs == 'y':
                # sets the maximum time to the last value of t_y
                t_max = int(t_y[-1])
                if custom_values == 'y':
                    # uses the custom values specified previously
                    v_a, y_a, t_v_a, t_y_a = analytical_predictions(
                        n, y0, v0, t_max, 0.0, mass, x_section_area, drag_coefficient, air_density)
                else:
                    v_a, y_a, t_v_a, t_y_a = analytical_predictions(
                        t_max=t_max)

                # checks if v >= v_sound at any point in the freefall (as it is
                # variable), and plots v_sound on the graph
                v_sound = np.zeros(len(y))
                v_sound_passed = 343
                for i in range(0, len(y)):
                    v_sound[i] = speed_of_sound(y[i])
                    if abs(v[i]) >= abs(v_sound[i]):
                        v_sound_passed = -1*v_sound[i]
                        plt.axhline(v_sound_passed, color='orange', linestyle='--',
                                    label='Speed of sound = {:.2f}m/s\n(at {:.1f}m) '.format(v_sound_passed, y[i]))
                        plt.legend()
                        break

                # plot comparison graphs
                plot_graphs(t_v, v, 'v', True, t_v_a, v_a)
                plt.show()
                plot_graphs(t_y, y, 'y', True, t_y_a, y_a)
                plt.show()
                break
            elif compare_graphs == 'n':
                # checks if v >= v_sound at any point in the freefall (as it is
                # variable), and plots v_sound on the graph
                v_sound = np.zeros(len(y))
                v_sound_passed = 343
                for i in range(0, len(y)):
                    v_sound[i] = speed_of_sound(y[i])
                    if abs(v[i]) >= abs(v_sound[i]):
                        v_sound_passed = -1*v_sound[i]
                        plt.axhline(v_sound_passed, color='orange', linestyle='--',
                                    label='Speed of sound = {:.2f}m/s\n(at {:.1f}m) '.format(v_sound_passed, y[i]))
                        plt.legend()
                        break

                # calls the plot_graphs function to handle pyplot settings
                plot_graphs(t_v, v, 'v')
                plt.show()
                plot_graphs(t_y, y, 'y')
                plt.show()
                break
            else:
                print("Invalid input. Please try again.")
                continue

    elif user_input == 'c':
        print("Solving the freefall of a body with varying air density.")
        # loop until valid input
        while True:
            try:
                custom_values = str(input(
                    "Would you like to specify the values used in the problem? (y/n) ")).lower()
            except ValueError:
                # not a string
                print("Invalid input. Please try again.")
                continue
            if custom_values == 'y':
                fn = 'euler'
                # calls _custom_values() for the euler solution to allow the
                # parameters used to be changed
                n, y0, v0, mass, x_section_area, drag_coefficient, air_density, t0, delta_t = _custom_values(
                    fn)

                # runs the euler method with varying air density
                v, y, t_v, t_y = euler(
                    n, y0, t0, delta_t, v0, mass, x_section_area, drag_coefficient, air_density, var_air_dens=True)
                break
            elif custom_values == 'n':
                print('Using default values.')
                v, y, t_v, t_y = euler(var_air_dens=True)
                break
            else:
                print("Invalid input. Please try again.")
                continue

        # plots graphs of velocity and height against time
        while True:
            try:
                compare_graphs = str(input(
                    "Would you like to compare the results with the Euler solution with fixed drag? (y/n) ")).lower()
            except ValueError:
                # not a string
                print("Invalid input. Please try again.")
                continue
            if compare_graphs == 'y':
                t_max = int(t_y[-1])
                if custom_values == 'y':
                    v_a, y_a, t_v_a, t_y_a = euler(
                        n, y0, t0, delta_t, v0, mass, x_section_area, drag_coefficient, air_density)
                else:
                    v_a, y_a, t_v_a, t_y_a = euler()

                # checks if v >= v_sound at any point in the freefall (as it is
                # variable), and plots v_sound on the graph
                v_sound = np.zeros(len(y))
                v_sound_passed = 343
                for i in range(0, len(y)):
                    v_sound[i] = speed_of_sound(y[i])
                    if abs(v[i]) >= abs(v_sound[i]):
                        v_sound_passed = -1*v_sound[i]
                        plt.axhline(v_sound_passed, color='orange', linestyle='--',
                                    label='Speed of sound = {:.2f}m/s\n(at {:.1f}m) '.format(v_sound_passed, y[i]))
                        plt.legend()
                        break

                # plot comparison graphs
                plot_graphs(t_v_a, v_a, 'v', True, t_v, v, 'Modified Euler')
                plt.show()
                plot_graphs(t_y_a, y_a, 'y', True, t_y, y, 'Modified Euler')
                plt.show()
                break
            elif compare_graphs == 'n':
                # checks if v >= v_sound at any point in the freefall (as it is
                # variable), and plots v_sound on the graph
                v_sound = np.zeros(len(y))
                v_sound_passed = 343
                for i in range(0, len(y)):
                    v_sound[i] = speed_of_sound(y[i])
                    if abs(v[i]) >= abs(v_sound[i]):
                        v_sound_passed = -1*v_sound[i]
                        plt.axhline(v_sound_passed, color='orange', linestyle='--',
                                    label='Speed of sound = {:.2f}m/s\n(at {:.1f}m) '.format(v_sound_passed, y[i]))
                        plt.legend()
                        break
                # calls the plot_graphs function to handle pyplot settings
                plot_graphs(t_v, v, 'v')
                plt.show()
                plot_graphs(t_y, y, 'y')
                plt.show()
                break
            else:
                print("Invalid input. Please try again.")
                continue

    # handle any other input
    elif user_input != 'q':
        print('Invalid input.')
