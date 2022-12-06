import numpy as np

from typing import List, Tuple


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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    TODO: docstring
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
    # TODO: these constants should be part of the loss calculation
    delta_h = (update_h @ previous_e) / 5
    # apply magnetic currents (sources)
    delta_h += source_h
    # update h (with loss: see notes)
    new_h = loss_h[0] * delta_h + loss_h[1] * previous_h
    # calculate dE/dt
    delta_e = (update_e @ new_h) / 10
    # apply electric currents (sources)
    delta_e += source_e
    # update e (with loss: see notes)
    new_e = loss_e[0] * delta_e + loss_e[1] * previous_e

    return new_h, new_e, delta_h, delta_e
