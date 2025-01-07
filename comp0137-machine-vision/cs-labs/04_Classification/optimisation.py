from itertools import count
import numpy as np
from numpy.linalg import lstsq
from scipy.optimize import fmin
from numerical import finite_difference


def line_search(start_position, update_direction, function, tolerance=1e-5):
    def line(offset):
        return function(start_position + offset * update_direction)

    offset = fmin(line, x0=0.0, xtol=tolerance, disp=False)
    return start_position + offset * update_direction


class Optimiser:
    def step(x, function):
        raise NotImplementedError()


class NewtonMethod(Optimiser):
    def step(x, function):
        J = function.jacobian(x)
        H = function.hessian(x)
        
        # this is equivalent to -inv(H) @ J, but more numerically stable
        update_direction = -lstsq(H, J, rcond=None)[0]

        return line_search(x, update_direction, function)


class SteepestDescent(Optimiser):
    # TODO
    # compute the gradient (Jacobian) of the function at the current position
    def step(x, function):
        gradient = function.jacobian(x)
        # determine the update direction at the negative gradient
        update_direction = -gradient 
        # determin the step size by line search
        return line_search(x, update_direction, function)


def array_from_dict(d):
    return np.stack([d[i] for i in sorted(d)])


def optimise(start_position, tolerance, function, optimiser, max_iterations=None):
    x = {0: start_position}
    w = {0: function(start_position)}

    for i in count(start=0):
        x[i + 1] = optimiser.step(x[i], function)
        w[i + 1] = function(x[i + 1])

        if all(abs(x[i + 1] - x[i]) < tolerance) or (max_iterations is not None and i >= max_iterations):
            break
        print('Iteration {:4d}, Function {}'.format(i + 1, w[i + 1]))

    return array_from_dict(x), array_from_dict(w)
