#%% imports
from typing import List, Tuple
import numpy as np

import fdtd_solver


def tfsf_2d_ex(
    previous_h: np.ndarray,
    previous_e: np.ndarray,
    update_h: np.ndarray,
    update_e: np.ndarray,
    source_e: np.ndarray,
    loss_h: List[np.ndarray],
    loss_e: List[np.ndarray],
) -> Tuple[np.ndarray]:

    source_h = np.zeros_like(source_e)

    new_h, new_e, _, _ = fdtd_solver.step(
        previous_h,
        previous_e,
        update_h,
        update_e,
        source_h,
        source_e,
        loss_h,
        loss_e,
    )

    je = loss_h[0] * source_h
    jm = loss_e[0] * source_e

    return new_h, new_e, je, jm
