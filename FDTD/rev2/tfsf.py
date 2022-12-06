#%% imports
from typing import List, Tuple
import numpy as np

import fdtd_solver


def tfsf_2d_ex(
    previous_h: np.ndarray,
    previous_e: np.ndarray,
    update_h: np.ndarray,
    update_e: np.ndarray,
    source_h: np.ndarray,
    source_e: np.ndarray,
    loss_h: List[np.ndarray],
    loss_e: List[np.ndarray],
) -> Tuple[np.ndarray]:
    pass
