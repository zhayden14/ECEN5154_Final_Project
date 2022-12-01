#%% imports
import numpy as np


#%% PEC stencil
pec_naive_e = np.zeros((1, 1, 1, 3), dtype=float)
# ped_naive_h = np.zeros() do we need to set this?
pec_extents = [[0, 0], [0, 0], [0, 0], [0, 0]]

#%% Free space stencil
free_ex = [
    [
        [[0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0]],
    ],
    [
        [[0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0]],
    ],
]
free_ey = 0
free_ey = 0
free_hx = 0
free_hy = 0
free_hz = 0
free_extents_x = [[-1, 0], [-1, 0], [-1, 0], [0, 2]]
free_extents_y = [[-1, 0], [-1, 0], [-1, 0], [-1, 1]]
free_extents_z = [[-1, 0], [-1, 0], [-1, 0], [-2, 0]]

# %%
