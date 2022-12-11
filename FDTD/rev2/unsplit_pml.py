"""
unsplit_pml.py
Zachary Hayden
ECEN5154, Fall 2022

Perfectly matched layer implementation
"""

#%%
import numpy as np

import fd_stencils


def pml_coefficients(
    a_x, a_y, a_z, kappa_x, kappa_y, kappa_z, cond_x, cond_y, cond_z, delta_t
):
    """Calculates coefficents for use in PML calculations"""
    EPSILON_0 = 8.854e-12
    # b_w terms
    b_x = np.exp(-((a_x / EPSILON_0) + (cond_x / EPSILON_0)) * delta_t)
    b_y = np.exp(-((a_y / EPSILON_0) + (cond_y / EPSILON_0)) * delta_t)
    b_z = np.exp(-((a_z / EPSILON_0) + (cond_z / EPSILON_0)) * delta_t)
    # C_w terms
    c_x = cond_x / (cond_x * kappa_x + np.power(kappa_x, 2) * a_x) * (b_x - 1)
    c_y = cond_y / (cond_y * kappa_y + np.power(kappa_y, 2) * a_y) * (b_y - 1)
    c_z = cond_y / (cond_z * kappa_z + np.power(kappa_z, 2) * a_z) * (b_z - 1)

    return (b_x, b_y, b_z), (c_x, c_y, c_z)


def pml_stencils_2d(c_w, kappa_w):
    """Defines 2D stencils to generate finite differences for PML calculations"""
    c_x = c_w[0]
    c_y = c_w[1]
    c_z = c_w[2]
    kappa_x = kappa_w[0]
    kappa_y = kappa_w[1]
    kappa_z = kappa_w[2]

    definition = {
        "ex_pml": {
            "points": [(0, 0, 0, 0), (0, -1, 0, 0)],
            "values": [1 / kappa_x, -1 / kappa_x],
        },
        "ey_pml": {
            "points": [(0, 0, 0, -1), (0, 0, -1, -1)],
            "values": [-1 / kappa_y, 1 / kappa_y],
        },
        "hz_pml": {
            "points": [(0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 1, 1), (0, 0, 0, 1)],
            "values": [1 / kappa_z, -1 / kappa_z, -1 / kappa_z, 1 / kappa_z],
        },
        "ex_pml_update": {
            "points": [(0, 0, 0, 0), (0, -1, 0, 0)],
            "values": [c_x, -c_x],
        },
        "ey_pml_update": {
            "points": [(0, 0, 0, -1), (0, 0, -1, -1)],
            "values": [-c_y, c_y],
        },
        "hz_pml_x_update": {
            "points": [(0, 0, 1, 1), (0, 0, 0, 1)],
            "values": [-c_z, c_z],
        },
        "hz_pml_y_update": {
            "points": [(0, 1, 0, 0), (0, 0, 0, 0)],
            "values": [c_z, -c_z],
        },
    }
    names = []
    stencils = []
    extents = []
    for name in definition.keys():
        names.append(name)
        stencil, extent = fd_stencils.points_to_stencil(
            definition[name]["points"], definition[name]["values"]
        )
        stencils.append(stencil)
        extents.append(extent)

    return names, stencils, extents


def pml_stencils_1d(c_w, kappa_w):
    """Defines 1D stencils to generate finite differences for PML calculations"""
    c_x = c_w[0]
    c_z = c_w[2]
    kappa_x = kappa_w[0]
    kappa_z = kappa_w[2]

    definition = {
        "ex_pml": {
            "points": [(0, 0, 0, 0), (0, -1, 0, 0)],
            "values": [1 / kappa_x, -1 / kappa_x],
        },
        "hz_pml": {
            "points": [(0, 1, 0, 0), (0, 0, 0, 0)],
            "values": [1 / kappa_z, -1 / kappa_z],
        },
        "ex_pml_update": {
            "points": [(0, 0, 0, 0), (0, -1, 0, 0)],
            "values": [c_x, -c_x],
        },
        "hz_pml_update": {
            "points": [(0, 1, 0, 0), (0, 0, 0, 0)],
            "values": [c_z, -c_z],
        },
    }

    names = []
    stencils = []
    extents = []
    for name in definition.keys():
        names.append(name)
        stencil, extent = fd_stencils.points_to_stencil(
            definition[name]["points"], definition[name]["values"]
        )
        stencils.append(stencil)
        extents.append(extent)

    return names, stencils, extents


def update_pml_e(pml_stencil, pml_convolution, b_w, h):
    """Updates E-field convolution for PML calculations"""
    # NOTE: create b_w vector at top level
    pml_convolution = (pml_stencil @ h) + (b_w * pml_convolution)

    return pml_convolution


def update_pml_h(pml_stencil, pml_convolution, b_w, e):
    """Updates H-field convolution for PML calculations"""
    # NOTE: create b_w vector at top level
    # NOTE: approximate e as 32-bit floats to reduce memory usage
    e_32 = np.array(e, dtype=np.float32)
    pml_convolution = (pml_stencil @ e_32) + (b_w * pml_convolution)

    return pml_convolution
