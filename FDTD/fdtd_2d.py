#%% imports
import numpy as np

import fd_lib, fdtd_solver


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
# E-field dimensions: z_dim (1), y_dim, x_dim, component

# let's start with a 16x16 grid
e_fields = np.zeros((1, 16, 16, 2), dtype=float)
h_fields = np.zeros((1, 16, 16, 1), dtype=float)

e_lossless = [
    np.ones((1, 16, 16, 2), dtype=float),
    np.zeros((1, 16, 16, 2), dtype=float),
]
h_lossless = [
    np.ones((1, 16, 16, 1), dtype=float),
    np.zeros((1, 16, 16, 1), dtype=float),
]

e_sources = np.zeros((1, 16, 16, 2), dtype=float)
e_sources[0, 8, 8, 0] = 1.0
h_sources = np.zeros((1, 16, 16, 1), dtype=float)

ex_const_stencil = np.array([[[[0]]]])
ex_const_extents = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
ey_const_stencil = np.array([[[[0]]]])
ey_const_extents = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
h_const_stencil = np.array([[[[0]]]])
h_const_extents = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])

ex_free_stencil = np.array([[[[-1]]], [[[1]]]])
ex_free_extents = np.array([[0, 0], [-1, 0], [0, 0], [0, 0]])
ey_free_stencil = np.array([[[[1]], [[-1]]]])
ey_free_extents = np.array([[0, 0], [0, 0], [-1, 0], [0, 0]])
h_free_stencil = np.array([[[[0], [0]], [[-1], [0]]], [[[0], [1]], [[1], [-1]]]])
h_free_extents = np.array([[0, 0], [-1, 0], [0, 0], [0, 1]])

e_system_matrix = np.zeros((1,16,16,1), dtype=float)
e_index_matrix = 
h_system_matrix = np.zeros((1,16,16,2), dtype=float)
h_index_matrix = 
