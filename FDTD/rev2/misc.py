import itertools
import numpy as np


def generate_gradient(shape: tuple, deltas: tuple):

    gradient = np.zeros(shape=shape)

    coordinates = [np.arange(0, shape[dim], 1) for dim in range(len(shape))]
    values = [np.arange(0, shape[dim], 1) * deltas[dim] for dim in range(len(shape))]
    for entry in itertools.product(*tuple(coordinates)):
        value = 0
        for dim in range(len(shape)):
            value += values[dim][entry[dim]]
        gradient[entry] = value

    return gradient


if __name__ == "__main__":
    # simple 2D case: horizontal gradient
    gradient_2d_horizontal = generate_gradient((8, 8), (0, 1))
    # complex 3D case: two-axis gradient
    gradient_2d_both = generate_gradient((10, 10), (0.5, -0.1))
    # representative case: 4D with two-axis gradient
    fdtd_gradient = generate_gradient((1, 8, 8, 2), (0, 0.125, -0.250, 0))
    pass
