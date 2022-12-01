#%% imports
import numpy as np
import matplotlib.pyplot as plt

import fd_lib, fdtd_solver


# %% basic stencils
# E-field dimensions: z_dim (1), y_dim, x_dim, component

ex_const_stencil = np.array([[[[0]]]])
ex_const_extents = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
ey_const_stencil = np.array([[[[0]]]])
ey_const_extents = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
hz_const_stencil = np.array([[[[0]]]])
hz_const_extents = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])

ex_free_stencil = np.array([[[[-0.5]], [[0.5]]]])
ex_free_extents = np.array([[0, 0], [-1, 0], [0, 0], [0, 0]])
ey_free_stencil = np.array([[[[0.5], [-0.5]]]])
ey_free_extents = np.array([[0, 0], [0, 0], [-1, 0], [-1, -1]])
hz_free_stencil = np.array([[[[0, 0], [-0.5, 0]], [[0, 0.5], [0.5, -0.5]]]])
hz_free_extents = np.array([[0, 0], [-1, 0], [-1, 0], [0, 1]])

stencil_list = [
    ex_const_stencil,
    ey_const_stencil,
    hz_const_stencil,
    ex_free_stencil,
    ey_free_stencil,
    hz_free_stencil,
]
extent_list = [
    ex_const_extents,
    ey_const_extents,
    hz_const_extents,
    ex_free_extents,
    ey_free_extents,
    hz_free_extents,
]
stencil_names = {
    "ex_const": 0,
    "ey_const": 1,
    "hz_const": 2,
    "ex_free": 3,
    "ey_free": 4,
    "hz_free": 5,
}

#%% start setting up variables
# let's start with a 16x16 grid

# TODO: parameterize e and h shape?

X_NPOINTS = 16
Y_NPOINTS = 16
Z_NPOINTS = 1
E_COMPONENTS = 2
E_TOTAL = X_NPOINTS * Y_NPOINTS * Z_NPOINTS * E_COMPONENTS
H_COMPONENTS = 1
H_TOTAL = X_NPOINTS * Y_NPOINTS * Z_NPOINTS * H_COMPONENTS

e_shape = (Z_NPOINTS, Y_NPOINTS, X_NPOINTS, E_COMPONENTS)
h_shape = (Z_NPOINTS, Y_NPOINTS, X_NPOINTS, H_COMPONENTS)

""" indexing and calculation matrices"""
e_index_matrix = fd_lib.node_index_matrix(e_shape)
h_index_matrix = fd_lib.node_index_matrix(h_shape)

e_system_matrix = np.zeros(
    (np.max(e_index_matrix) + 1, np.max(h_index_matrix) + 1), dtype=float
)
h_system_matrix = np.zeros(
    (np.max(h_index_matrix) + 1, np.max(e_index_matrix) + 1), dtype=float
)

e_flag_matrix = np.zeros(e_shape, dtype=int)
h_flag_matrix = np.zeros(h_shape, dtype=int)

"""physical values (more or less)"""
e_fields = np.zeros(e_shape, dtype=float)
# e_fields[0, 8, 8, 0] = 1.0
h_fields = np.zeros(h_shape, dtype=float)

e_lossless = [
    np.ones(e_shape, dtype=float).reshape((E_TOTAL, 1)),
    np.zeros(e_shape, dtype=float).reshape((E_TOTAL, 1)),
]
h_lossless = [
    np.ones(h_shape, dtype=float).reshape((H_TOTAL, 1)),
    np.zeros(h_shape, dtype=float).reshape((H_TOTAL, 1)),
]

e_sources = np.zeros(e_shape, dtype=float)
e_sources[0, int(Y_NPOINTS / 2), int(X_NPOINTS / 2), 0] = 1.0
h_sources = np.zeros(h_shape, dtype=float)

#%% apply flags manually for now
e_flag_matrix[0, :, :, 0] = stencil_names["ex_free"]
e_flag_matrix[0, :, :, 1] = stencil_names["ey_free"]
e_flag_matrix[0, 0, :, 0] = stencil_names["ex_const"]
e_flag_matrix[0, -1, :, 0] = stencil_names["ex_const"]
e_flag_matrix[0, :, 0, 0] = stencil_names["ex_const"]
e_flag_matrix[0, :, -1, 0] = stencil_names["ex_const"]
e_flag_matrix[0, 0, :, 1] = stencil_names["ey_const"]
e_flag_matrix[0, -1, :, 1] = stencil_names["ey_const"]
e_flag_matrix[0, :, 0, 1] = stencil_names["ey_const"]
e_flag_matrix[0, :, -1, 1] = stencil_names["ey_const"]

h_flag_matrix[0, :, :, 0] = stencil_names["hz_free"]
h_flag_matrix[0, 0, :, 0] = stencil_names["hz_const"]
h_flag_matrix[0, -1, :, 0] = stencil_names["hz_const"]
h_flag_matrix[0, :, 0, 0] = stencil_names["hz_const"]
h_flag_matrix[0, :, -1, 0] = stencil_names["hz_const"]

# %% fill system matrices
# E-field
fd_lib.apply_stencil(
    system_matrix=e_system_matrix,
    flag_matrix=e_flag_matrix,
    index_matrix_rows=e_index_matrix,
    index_matrix_columns=h_index_matrix,
    stencils=stencil_list,
    stencil_extents=extent_list,
    domain_indices=[
        np.arange(0, Z_NPOINTS, 1),
        np.arange(0, Y_NPOINTS, 1),
        np.arange(0, X_NPOINTS, 1),
        np.arange(0, E_COMPONENTS, 1),
    ],
)
# H-field
fd_lib.apply_stencil(
    system_matrix=h_system_matrix,
    flag_matrix=h_flag_matrix,
    index_matrix_rows=h_index_matrix,
    index_matrix_columns=e_index_matrix,
    stencils=stencil_list,
    stencil_extents=extent_list,
    domain_indices=[
        np.arange(0, Z_NPOINTS, 1),
        np.arange(0, Y_NPOINTS, 1),
        np.arange(0, X_NPOINTS, 1),
        np.arange(0, H_COMPONENTS, 1),
    ],
)

# %%
# TODO: make this programmatic
e_field_vector = e_fields.reshape((512, 1))
h_field_vector = h_fields.reshape((256, 1))
e_sources_amp_vector = e_sources.reshape((512, 1))
h_sources_vector = h_sources.reshape((256, 1))
for i in range(32):
    e_sources_vector = e_sources_amp_vector * np.sin(np.pi / 8 * i)
    h_field_vector, e_field_vector = fdtd_solver.step(
        previous_h=h_field_vector,
        previous_e=e_field_vector,
        update_h=h_system_matrix,
        update_e=e_system_matrix,
        source_h=h_sources_vector,
        source_e=e_sources_vector,
        loss_h=h_lossless,
        loss_e=e_lossless,
    )
    # TODO: subplots
    plt.figure()
    plt.pcolormesh(
        e_field_vector.reshape(1, 16, 16, 2)[0, :, :, 0],
        cmap=plt.cm.turbo,
        vmin=-2.0,
        vmax=2.0,
    )
    plt.colorbar()
    plt.title(f"Ex Fields")
    plt.figure()
    plt.pcolormesh(
        e_field_vector.reshape(1, 16, 16, 2)[0, :, :, 1],
        cmap=plt.cm.turbo,
        vmin=-2.0,
        vmax=2.0,
    )
    plt.colorbar()
    plt.title(f"Ey Fields")
    plt.figure()
    plt.pcolormesh(
        h_field_vector.reshape(1, 16, 16, 1)[0, :, :, 0],
        cmap=plt.cm.turbo,
        vmin=-2.0,
        vmax=2.0,
    )
    plt.colorbar()
    plt.title(f"Hz Fields")
    plt.show()

# %%
