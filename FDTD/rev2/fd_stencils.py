# %%
from typing import List
import itertools
import numpy as np

# %%
# stencils: (z, y, x, component)


def points_to_stencil(points: List[tuple], values: List[tuple]):
    # TODO: user-defined extents?
    # calculate extents
    extent_array = np.array([point for point in points])
    extent_min = np.min(extent_array, axis=0)
    extent_max = np.max(extent_array, axis=0)
    shape = extent_max - extent_min + 1
    # create a blank stencil
    stencil = np.zeros(shape=shape)
    # fill the stencil
    for point, value in itertools.zip_longest(points, values):
        stencil[tuple(point - extent_min)] = value
    # format extents
    extents = np.array([extent_min, extent_max]).transpose()
    return stencil, extents


def stencils_2d_free_space():
    definition = {
        "pec": {"points": [(0, 0, 0, 0)], "values": [0.0]},
        # apply programmatically to Ex when node @-y is PEC
        "ex_pec_border": {"points": [(0, 0, 0, 0)], "values": [0.0]},
        # apply programmatically to Ey when node @-x is PEC
        "ey_pec_border": {"points": [(0, 0, 0, 0)], "values": [0.0]},
        "ex_free": {"points": [(0, 0, 0, 0), (0, 1, 0, 0)], "values": [-0.5, 0.5]},
        "ey_free": {"points": [(0, 0, -1, -1), (0, 0, 0, -1)], "values": [0.5, -0.5]},
        "hz_free": {
            "points": [(0, 0, 0, 0), (0, 1, 0, 0), (0, 0, -1, 1), (0, 0, 0, 1)],
            "values": [-0.5, 0.5, 0.5, -0.5],
        },
    }
    names = []
    stencils = []
    extents = []
    for name in definition.keys():
        names.append(name)
        stencil, extent = points_to_stencil(
            definition[name]["points"], definition[name]["values"]
        )
        stencils.append(stencil)
        extents.append(extent)

    return names, stencils, extents


# %% old stencils
ex_const_stencil = np.array([[[[0]]]])
ex_const_extents = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
ey_const_stencil = np.array([[[[0]]]])
ey_const_extents = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
hz_const_stencil = np.array([[[[0]]]])
hz_const_extents = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])

ex_free_stencil = np.array([[[[-0.5]], [[0.5]]]])
ex_free_extents = np.array([[0, 0], [0, 1], [0, 0], [0, 0]])
ey_free_stencil = np.array([[[[0.5], [-0.5]]]])
ey_free_extents = np.array([[0, 0], [0, 0], [-1, 0], [-1, -1]])
hz_free_stencil = np.array([[[[0, 0], [-0.5, 0]], [[0, 0.5], [0.5, -0.5]]]])
hz_free_extents = np.array([[0, 0], [-1, 0], [0, 1], [0, 1]])

# %%
