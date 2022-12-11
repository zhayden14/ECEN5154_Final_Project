"""
fdtd_2d_scratchpad.py
Zachary Hayden
ECEN5154, Fall 2022

2D simulation for testing/development of 2D wave propagation
"""

#%% imports
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import fd_lib, fdtd_solver, misc, fd_stencils, unsplit_pml

#%% Mesh
X_NPOINTS = 64
Y_NPOINTS = 64
Z_NPOINTS = 1
E_COMPONENTS = 2
E_TOTAL = X_NPOINTS * Y_NPOINTS * Z_NPOINTS * E_COMPONENTS
H_COMPONENTS = 1
H_TOTAL = X_NPOINTS * Y_NPOINTS * Z_NPOINTS * H_COMPONENTS

e_shape = (Z_NPOINTS, Y_NPOINTS, X_NPOINTS, E_COMPONENTS)
h_shape = (Z_NPOINTS, Y_NPOINTS, X_NPOINTS, H_COMPONENTS)

# temporal and spatial steps
delta_t = 0.1 * (0.1 / 3e8)
delta_x = 0.1

# %% PML constants
kappa_w = (4, 4, 4)
a_w = (1e-8, 1e-8, 1e-8)
cond_w = (2e-2, 2e-2, 2e-2)

b_w, c_w = unsplit_pml.pml_coefficients(
    a_w[0],
    a_w[1],
    a_w[2],
    kappa_w[0],
    kappa_w[1],
    kappa_w[2],
    cond_w[0],
    cond_w[1],
    cond_w[2],
    delta_t,
)

#%% use new stencil capabilities
names, stencil_list, extent_list = fd_stencils.stencils_2d_free_space()
stencil_names = {f"{names[i]}": i for i in range(len(names))}

pml_names, pml_stencils, pml_extents = unsplit_pml.pml_stencils_2d(c_w, kappa_w)
for stencil in pml_stencils:
    stencil_list.append(stencil)
for extent in pml_extents:
    extent_list.append(extent)
prev_count = len(stencil_names.keys())
for idx, name in enumerate(pml_names):
    stencil_names[name] = prev_count + idx

#%% start setting up variables
# let's start with a 16x16 grid

""" indexing and calculation matrices"""
e_index_matrix = fd_lib.node_index_matrix(e_shape)
h_index_matrix = fd_lib.node_index_matrix(h_shape)

e_system_matrix = np.zeros(
    (np.max(e_index_matrix) + 1, np.max(h_index_matrix) + 1), dtype=float
)
e_pml_system_matrix = np.zeros_like(e_system_matrix)
h_system_matrix = np.zeros(
    (np.max(h_index_matrix) + 1, np.max(e_index_matrix) + 1), dtype=float
)
# NOTE: size of E is a hacky way to define sizes for H PML matrices
# NOTE: lower precision to get around numpy limitations
h_pml_system_matrix = np.zeros(
    (np.max(e_index_matrix) + 1, np.max(e_index_matrix) + 1), dtype=np.float32
)

e_flag_matrix = np.zeros(e_shape, dtype=int)
e_flag_pml = np.zeros_like(e_flag_matrix)
h_flag_matrix = np.zeros(h_shape, dtype=int)
h_flag_pml = np.zeros_like(e_flag_matrix)

"""physical values (more or less)"""
e_fields = np.zeros(e_shape, dtype=float)
# e_fields = misc.generate_gradient(e_shape, (0, 0, 1, 0))
# e_fields[0, 8, 8, :] = 1.0
h_fields = np.zeros(h_shape, dtype=float)
# h_fields = misc.generate_gradient(h_shape, (0, 0, 2, 0))
# h_fields[0, 8, 8, 0] = 1.0

c_e = delta_t / (4e-6 * np.pi * delta_x)
c_h = delta_t / (8.854e-12 * delta_x)

e_lossless = [
    np.ones(e_shape, dtype=float).reshape((E_TOTAL, 1)),
    np.ones(e_shape, dtype=float).reshape((E_TOTAL, 1)),
]
h_lossless = [
    np.ones(h_shape, dtype=float).reshape((H_TOTAL, 1)),
    np.ones(h_shape, dtype=float).reshape((H_TOTAL, 1)),
]

e_sources = np.zeros(e_shape, dtype=float)
e_sources[0, 20, 16:-16, 0] = 1.0
# e_sources[0, :, 4, 0] = 1.0
# e_sources[0, int(Y_NPOINTS / 2), int(X_NPOINTS / 2), 0] = 1.0
# e_sources[0, int(Y_NPOINTS / 2), int(X_NPOINTS / 2) + 1, 0] = -1.0
h_sources = np.zeros(h_shape, dtype=float)

#%% apply flags manually for now
e_flag_matrix[0, :, :, 0] = stencil_names["ex_free"]
e_flag_matrix[0, :, :, 1] = stencil_names["ey_free"]
# e_flag_matrix[0, int(Y_NPOINTS / 4) : int(Y_NPOINTS / 4) + 3, 6:10, :] = stencil_names[
#     "pec"
# ]
e_flag_matrix[0, 0, :, :] = stencil_names["pec"]
e_flag_matrix[0, -1, :, :] = stencil_names["pec"]
e_flag_matrix[0, :, 0, :] = stencil_names["pec"]
e_flag_matrix[0, :, -1, :] = stencil_names["pec"]
e_flag_matrix[0, 1:15, 1:-1, 0] = stencil_names["ex_pml"]
e_flag_matrix[0, -15:-1, 1:-1, 0] = stencil_names["ex_pml"]
e_flag_matrix[0, 1:-1, 1:15, 0] = stencil_names["ex_pml"]
e_flag_matrix[0, 1:-1, -15:-1, 0] = stencil_names["ex_pml"]
e_flag_matrix[0, 1:15, 1:-1, 1] = stencil_names["ey_pml"]
e_flag_matrix[0, -15:-1, 1:-1, 1] = stencil_names["ey_pml"]
e_flag_matrix[0, 1:-1, 1:15, 1] = stencil_names["ey_pml"]
e_flag_matrix[0, 1:-1, -15:-1, 1] = stencil_names["ey_pml"]
#
e_flag_pml[0, 1:15, 1:-1, 0] = stencil_names["ex_pml_update"]
e_flag_pml[0, -15:-1, 1:-1, 0] = stencil_names["ex_pml_update"]
e_flag_pml[0, 1:-1, 1:15, 0] = stencil_names["ex_pml_update"]
e_flag_pml[0, 1:-1, -15:-1, 0] = stencil_names["ex_pml_update"]
e_flag_pml[0, 1:15, 1:-1, 1] = stencil_names["ey_pml_update"]
e_flag_pml[0, -15:-1, 1:-1, 1] = stencil_names["ey_pml_update"]
e_flag_pml[0, 1:-1, 1:15, 1] = stencil_names["ey_pml_update"]
e_flag_pml[0, 1:-1, -15:-1, 1] = stencil_names["ey_pml_update"]

h_flag_matrix[0, :, :, 0] = stencil_names["hz_free"]
h_flag_matrix[0, 0, :, :] = stencil_names["pec"]
h_flag_matrix[0, -1, :, :] = stencil_names["pec"]
h_flag_matrix[0, :, 0, :] = stencil_names["pec"]
h_flag_matrix[0, :, -1, :] = stencil_names["pec"]
h_flag_matrix[0, 1:15, 1:-1, 0] = stencil_names["hz_pml"]
h_flag_matrix[0, -15:-1, 1:-1, 0] = stencil_names["hz_pml"]
h_flag_matrix[0, 1:-1, 1:15, 0] = stencil_names["hz_pml"]
h_flag_matrix[0, 1:-1, -15:-1, 0] = stencil_names["hz_pml"]

#
h_flag_pml[0, 1:15, 1:-1, 0] = stencil_names["hz_pml_x_update"]
h_flag_pml[0, -15:-1, 1:-1, 0] = stencil_names["hz_pml_x_update"]
h_flag_pml[0, 1:-1, 1:15, 0] = stencil_names["hz_pml_x_update"]
h_flag_pml[0, 1:-1, -15:-1, 0] = stencil_names["hz_pml_x_update"]
h_flag_pml[0, 1:15, 1:-1, 1] = stencil_names["hz_pml_y_update"]
h_flag_pml[0, -15:-1, 1:-1, 1] = stencil_names["hz_pml_y_update"]
h_flag_pml[0, 1:-1, 1:15, 1] = stencil_names["hz_pml_y_update"]
h_flag_pml[0, 1:-1, -15:-1, 1] = stencil_names["hz_pml_y_update"]

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
fd_lib.apply_stencil(
    system_matrix=e_pml_system_matrix,
    flag_matrix=e_flag_pml,
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
fd_lib.apply_stencil(
    system_matrix=h_pml_system_matrix,
    flag_matrix=h_flag_pml,
    index_matrix_rows=e_index_matrix,
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
e_field_vector = e_fields.reshape((E_TOTAL, 1))
h_field_vector = h_fields.reshape((H_TOTAL, 1))
e_pml_convolution = np.zeros_like(e_field_vector)
h_pml_convolution = np.zeros_like(e_field_vector)
e_sources_amp_vector = e_sources.reshape((E_TOTAL, 1))
h_sources_vector = h_sources.reshape((H_TOTAL, 1))
e_data_vector = []
h_data_vector = []
e_data_delta = []
h_data_delta = []

# b_w defined here
b_w_e = np.zeros_like(e_flag_pml)
b_w_e[:, :, :, 0] = b_w[1]
b_w_e[:, :, :, 1] = -b_w[0]
b_w_e = np.reshape(b_w_e, (E_TOTAL, 1))
b_w_h = np.zeros_like(h_flag_pml)
b_w_h[:, :, :, 0] = -b_w[2]
b_w_h[:, :, :, 1] = b_w[2]
b_w_h = np.reshape(b_w_h, (E_TOTAL, 1))

for i in range(1024):
    # def update(i):
    if i < 17:
        e_sources_vector = e_sources_amp_vector * np.sin(np.pi / 8 * i)
    else:
        e_sources_vector = np.zeros_like(e_sources_vector)
    (
        h_field_vector,
        e_field_vector,
        h_delta,
        e_delta,
        h_pml_convolution,
        e_pml_convolution,
    ) = fdtd_solver.step(
        previous_h=h_field_vector,
        previous_e=e_field_vector,
        update_h=h_system_matrix,
        update_e=e_system_matrix,
        source_h=h_sources_vector,
        source_e=e_sources_vector,
        loss_h=h_lossless,
        loss_e=e_lossless,
        pml_convolution_h=h_pml_convolution,
        pml_convolution_e=e_pml_convolution,
        pml_stencil_h=h_pml_system_matrix,
        pml_stencil_e=e_pml_system_matrix,
        b_w_h=b_w_h,
        b_w_e=b_w_e,
    )
    e_data_vector.append(e_field_vector)
    h_data_vector.append(h_field_vector)
    e_data_delta.append(e_delta)
    h_data_delta.append(h_delta)
fig, axes = plt.subplots(2, 3)
# fig, axes = plt.subplots(2, 3)
# Ex
def update(e_data_vector, h_data_vector, e_data_delta, h_data_delta, i):
    i = i % len(e_data_vector)
    ax = axes[0, 0]
    pcm = ax.pcolormesh(
        e_data_vector[i].reshape(1, Y_NPOINTS, X_NPOINTS, 2)[0, :, :, 0],
        cmap=plt.cm.twilight,
        vmin=-2.0,
        vmax=2.0,
    )
    # fig.colorbar(pcm, ax=ax)
    ax.set_title(f"Ex Fields")
    # Ey
    ax = axes[0, 1]
    pcm = ax.pcolormesh(
        e_data_vector[i].reshape(1, Y_NPOINTS, X_NPOINTS, 2)[0, :, :, 1],
        cmap=plt.cm.twilight,
        vmin=-2.0,
        vmax=2.0,
    )
    # fig.colorbar(pcm, ax=ax)
    ax.set_title(f"Ey Fields")
    # Hz
    ax = axes[0, 2]
    pcm = ax.pcolormesh(
        h_data_vector[i].reshape(1, Y_NPOINTS, X_NPOINTS, 1)[0, :, :, 0],
        cmap=plt.cm.twilight,
        vmin=-2.0,
        vmax=2.0,
    )
    # fig.colorbar(pcm, ax=ax)
    ax.set_title(f"Hz Fields")
    # Ex
    ax = axes[1, 0]
    pcm = ax.pcolormesh(
        e_data_delta[i].reshape(1, Y_NPOINTS, X_NPOINTS, 2)[0, :, :, 0],
        cmap=plt.cm.twilight,
        vmin=-2.0,
        vmax=2.0,
    )
    # fig.colorbar(pcm, ax=ax)
    ax.set_title(f"Ex Delta")
    # Ey
    ax = axes[1, 1]
    pcm = ax.pcolormesh(
        e_data_delta[i].reshape(1, Y_NPOINTS, X_NPOINTS, 2)[0, :, :, 1],
        cmap=plt.cm.twilight,
        vmin=-2.0,
        vmax=2.0,
    )
    # fig.colorbar(pcm, ax=ax)
    ax.set_title(f"Ey Delta")
    # Hz
    ax = axes[1, 2]
    pcm = ax.pcolormesh(
        h_data_delta[i].reshape(1, Y_NPOINTS, X_NPOINTS, 1)[0, :, :, 0],
        cmap=plt.cm.twilight,
        vmin=-2.0,
        vmax=2.0,
    )
    # fig.colorbar(pcm, ax=ax)
    ax.set_title(f"Hz Delta")
    # plt.show()

    # plt.figure()
    # plt.pcolormesh(
    #     e_field_vector.reshape(1, 16, 16, 2)[0, :, :, 0],
    #     cmap=plt.cm.turbo,
    #     vmin=-2.0,
    #     vmax=2.0,
    # )
    # plt.colorbar()
    # plt.title(f"Ex Fields")
    # plt.figure()
    # plt.pcolormesh(
    #     e_field_vector.reshape(1, 16, 16, 2)[0, :, :, 1],
    #     cmap=plt.cm.turbo,
    #     vmin=-2.0,
    #     vmax=2.0,
    # )
    # plt.colorbar()
    # plt.title(f"Ey Fields")
    # plt.figure()
    # plt.pcolormesh(
    #     h_field_vector.reshape(1, 16, 16, 1)[0, :, :, 0],
    #     cmap=plt.cm.turbo,
    #     vmin=-2.0,
    #     vmax=2.0,
    # )
    # plt.colorbar()
    # plt.title(f"Hz Fields")
    # plt.show()


animation = FuncAnimation(
    fig,
    partial(update, e_data_vector, h_data_vector, e_data_delta, h_data_delta),
    interval=10,
    frames=1024,
    repeat=True,
)
plt.show()

# %%
