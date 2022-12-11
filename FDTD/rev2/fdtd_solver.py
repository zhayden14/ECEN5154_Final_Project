"""
fdtd_solver.py
Zachary Hayden
ECEN5154, Fall 2022

FDTD update function. Calculates the next E and H field values 
and applies sources and PML boundary condition.
"""

import numpy as np

from typing import List, Tuple

import unsplit_pml


# TODO: rename
def step(
    previous_h: np.ndarray,
    previous_e: np.ndarray,
    update_h: np.ndarray,
    update_e: np.ndarray,
    source_h: np.ndarray,
    source_e: np.ndarray,
    loss_h: List[np.ndarray],
    loss_e: List[np.ndarray],
    pml_convolution_h: np.ndarray,
    pml_convolution_e: np.ndarray,
    pml_stencil_h: np.ndarray,
    pml_stencil_e: np.ndarray,
    b_w_h: np.ndarray,
    b_w_e: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Updates the E and H fields, given previous E and H fields, sources, PML data and coefficients for application of
    conduction currents
    """
    # # "equivalent" of system matrix, multiply this by other field to get source-free delta
    # update_h = np.zeros()
    # update_e = np.zeros()

    # # sources (currents)
    # # NOTE: apply scaling factor at next level up. this includes the negative sign
    # source_e = np.zeros()
    # source_h = np.zeros()

    # # loss. applied like stencil for delta plus previous values
    # # TODO: allow this to be extended to higher-order finite differences?
    # # form is [delta coeff., prev. coeff.]
    # loss_h = [np.zeros(), np.zeros()]
    # loss_e = [np.zeros(), np.zeros()]

    """propagate solution"""
    # calculate dH/dt
    delta_h = update_h @ previous_e
    # apply pml corrections to magnetic currents
    pml_convolution_h = unsplit_pml.update_pml_h(
        pml_stencil_h, pml_convolution_h, b_w_h, previous_e
    )
    # hacky way to maintain compatibility with 1D
    if pml_convolution_h.shape != delta_h.shape:
        delta_h += pml_convolution_h[slice(0, int(len(pml_convolution_h) / 2))]
        delta_h += pml_convolution_h[
            slice(int(len(pml_convolution_h) / 2), len(pml_convolution_h))
        ]
    else:
        delta_h += pml_convolution_h.sum(axis=3)
    # apply magnetic currents (sources)
    delta_h += source_h
    # update h (with loss: see notes)
    new_h = loss_h[0] * delta_h + loss_h[1] * previous_h
    # calculate dE/dt
    delta_e = update_e @ new_h
    # apply pml corrections to electric currents
    pml_convolution_e = unsplit_pml.update_pml_e(
        pml_stencil_e, pml_convolution_e, b_w_e, new_h
    )
    delta_e += pml_convolution_e
    # apply electric currents (sources)
    delta_e += source_e
    # update e (with loss: see notes)
    new_e = loss_e[0] * delta_e + loss_e[1] * previous_e

    return new_h, new_e, delta_h, delta_e, pml_convolution_h, pml_convolution_e
