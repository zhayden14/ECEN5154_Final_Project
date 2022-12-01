"""
Appendix B
Higher-level FD solver functions and calculated potentials

ECEN 5154 Homework 2
Zachary Hayden
"""

import itertools
import functools
from pathlib import Path
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

import fd_lib


def apply_border(
    system_matrix: np.ndarray,
    excitation_vector: np.ndarray,
    index_matrix: np.ndarray,
    stencils: np.ndarray,
    excitations: np.ndarray,
    flag_matrix: np.ndarray,
) -> None:
    """Applies 1x1 PEC BCs around the edge of the computational domain"""
    n_rows = flag_matrix.shape[0]
    n_cols = flag_matrix.shape[1]
    # top and bottom
    fd_lib.apply_stencil_1x1(
        system_matrix,
        excitation_vector,
        index_matrix,
        stencils[flag_matrix[[0, -1], :]],
        excitations[flag_matrix[[0, -1], :]],
        rows=[0, n_rows - 1],
        cols=np.arange(0, n_cols, 1),
    )
    # left and right
    fd_lib.apply_stencil_1x1(
        system_matrix,
        excitation_vector,
        index_matrix,
        stencils[flag_matrix[:, [0, -1]]],
        excitations[flag_matrix[:, [0, -1]]],
        rows=np.arange(0, n_rows, 1),
        cols=[0, n_cols - 1],
    )


def argnearest(array, value):
    """Returns the index of the array value that is closest to the requested value"""
    temp = np.abs(array - value)
    return np.argmin(temp)


# TODO: this needs a better name
def apply_by_mesh(
    flag_matrix: np.ndarray,
    x_points: np.ndarray,
    y_points: np.ndarray,
    stencil_names: np.ndarray,
    stencil: str,
    x_extents: tuple,
    y_extents: tuple,
) -> None:
    """sets flags in the flag array given the flag name and x,y extents within the mesh"""
    # force extents into ndarray
    x_extents = np.atleast_1d(x_extents)
    y_extents = np.atleast_1d(y_extents)
    # find indices
    # NOTE: the order of the indices is not guaranteed at this point
    x_index_limits = np.array(
        [
            argnearest(x_points, np.min(x_extents)),
            argnearest(x_points, np.max(x_extents)),
        ]
    )
    y_index_limits = np.array(
        [
            argnearest(y_points, np.min(y_extents)),
            argnearest(y_points, np.max(y_extents)),
        ]
    )
    # generate grid of all points
    x_indices = np.arange(np.min(x_index_limits), np.max(x_index_limits) + 1, 1)
    y_indices = np.arange(np.min(y_index_limits), np.max(y_index_limits) + 1, 1)
    x_grid, y_grid = np.meshgrid(x_indices, y_indices)
    # set values in flag matrix
    flag_matrix[y_grid, x_grid] = stencil_names[stencil]


def excitation_constant(**kwargs):
    return kwargs["constant"]


def excitation_variable(**kwargs):
    return kwargs["variable"]


# TODO: I want to find a convenient way to loop through the test params specified in the homework
if __name__ == "__main__":
    """Setup output file directory"""
    # NOTE: I think this only works if THIS file is the top-level script
    # but we check __name__ == "__main__", so that should always be the case
    this_file = Path(__file__)
    hw2_folder = this_file.parent
    timestamp = datetime.now().strftime("%b_%d_%Y_%H-%M-%S")
    data_folder = hw2_folder / f"FD_{timestamp}"
    os.mkdir(data_folder)
    """Variables common to all meshes"""
    # generate list of stencils
    stencil_names = {
        "continuous medium": 0,
        "PEC 0V": 1,
        "PEC 1V": 2,
        "dielectric top boundary": 3,
        "dielectric bottom boundary": 4,
    }
    stencils = [
        fd_lib.stencil_iface_dielectric(1, 1, 1, 1, 1, 1, 1, 1)[0],
        np.array([[1]]),
        # TODO: explicitly separate edge PEC from potential center PEC at 0V
        np.array([[1]]),
        # top dielectric boundary
        fd_lib.stencil_iface_dielectric(1, 1, 2.22, 2.22, 1, 1, 1, 1)[0],
        # bottom dielectric boundary
        fd_lib.stencil_iface_dielectric(2.22, 2.22, 1, 1, 1, 1, 1, 1)[0],
    ]
    stencil_extents = [
        np.array([[-1, +1], [-1, +1]]),
        np.array([[0, 0], [0, 0]]),
        np.array([[0, 0], [0, 0]]),
        np.array([[-1, +1], [-1, +1]]),
        np.array([[-1, +1], [-1, +1]]),
    ]
    # excitations = [
    #     functools.partial(excitation_constant, constant=0),
    #     functools.partial(excitation_constant, constant=0),
    #     functools.partial(excitation_constant, constant=1),
    #     functools.partial(excitation_constant, constant=0),
    #     functools.partial(excitation_constant, constant=0),
    # ]
    excitations = [
        functools.partial(excitation_constant, constant=0),
        functools.partial(excitation_constant, constant=0),
        excitation_variable,
        functools.partial(excitation_constant, constant=0),
        functools.partial(excitation_constant, constant=0),
    ]
    """Loop through limits and step sizes (each combination has a unique mesh)"""
    step_sizes = [0.1, 0.05]
    x_sizes = [(0, 1), (-0.1, 1.1)]
    center_widths = [0.2, 0.6, 0.8]
    dielectrics = [False, True]
    for step, limits, dielectric in itertools.product(step_sizes, x_sizes, dielectrics):
        """generate mesh"""
        # mesh #1: 0.1 step size, d/b = 0
        x_step = step
        y_step = -1 * step
        x_limits = limits  # same as index order
        y_limits = (1, 0)  # opposite index order
        x_points = np.linspace(
            x_limits[0], x_limits[1], int((x_limits[1] - x_limits[0]) / x_step) + 1
        )
        y_points = np.linspace(
            y_limits[0], y_limits[1], int((y_limits[1] - y_limits[0]) / y_step) + 1
        )
        x_mesh, y_mesh = np.meshgrid(x_points, y_points)
        n_rows = x_mesh.shape[0]
        n_cols = x_mesh.shape[1]
        """ loop through (center conductor) geometry"""
        for center_width in center_widths:
            """init new system matrix, etc. for each solution"""
            # indexing and system matrix
            index_matrix = fd_lib.node_index_matrix((n_rows, n_cols))
            system_matrix = np.zeros((n_rows * n_cols, n_rows * n_cols), dtype=float)
            excitation_vector = np.zeros((n_rows * n_cols), dtype=float)
            # generate "flag" matrix
            # initialize to stencil 0 (free space)
            flag_matrix = np.zeros((n_rows, n_cols), dtype=int)

            """ mesh- and geometry-indpendent flags"""
            # PEC outer boundary
            flag_matrix[0, :] = stencil_names["PEC 0V"]
            flag_matrix[-1, :] = stencil_names["PEC 0V"]
            flag_matrix[:, 0] = stencil_names["PEC 0V"]
            flag_matrix[:, -1] = stencil_names["PEC 0V"]

            """ We need to split these out to get C' and Z"""
            if dielectric is True:
                # top dielectric boundary
                apply_by_mesh(
                    flag_matrix,
                    x_points,
                    y_points,
                    stencil_names,
                    "dielectric top boundary",
                    (
                        np.min(x_points) + np.abs(x_step),
                        np.max(x_points) - np.abs(x_step),
                    ),
                    (0.6, 0.6),
                )

                # bottom dielectric boundary
                apply_by_mesh(
                    flag_matrix,
                    x_points,
                    y_points,
                    stencil_names,
                    "dielectric bottom boundary",
                    (
                        np.min(x_points) + np.abs(x_step),
                        np.max(x_points) - np.abs(x_step),
                    ),
                    (0.4, 0.4),
                )

            """ mesh-dependent flags"""
            # PEC slots
            if np.min(x_points) < 0:
                apply_by_mesh(
                    flag_matrix,
                    x_points,
                    y_points,
                    stencil_names,
                    "PEC 0V",
                    (-0.1, 0),
                    (0, 0.4),
                )
                apply_by_mesh(
                    flag_matrix,
                    x_points,
                    y_points,
                    stencil_names,
                    "PEC 0V",
                    (-0.1, 0),
                    (0.6, 1),
                )
                apply_by_mesh(
                    flag_matrix,
                    x_points,
                    y_points,
                    stencil_names,
                    "PEC 0V",
                    (1, 1.1),
                    (0, 0.4),
                )
                apply_by_mesh(
                    flag_matrix,
                    x_points,
                    y_points,
                    stencil_names,
                    "PEC 0V",
                    (1, 1.1),
                    (0.6, 1),
                )

            """ geometry-dependent flags"""
            # PEC conductor
            pec_left = 0.5 - center_width / 2
            pec_right = 0.5 + center_width / 2
            apply_by_mesh(
                flag_matrix,
                x_points,
                y_points,
                stencil_names,
                "PEC 1V",
                (pec_left, pec_right),
                (0.6, 0.6),
            )

            """ Fill in system matrix"""
            fd_lib.apply_stencil_new(
                system_matrix=system_matrix,
                flag_matrix=flag_matrix,
                index_matrix=index_matrix,
                stencils=stencils,
                stencil_extents=stencil_extents,
                domain_indices=[
                    np.arange(0, n_rows, 1),
                    np.arange(0, n_cols, 1),
                ],
            )
            """ Fill in excitation vector"""
            fd_lib.apply_excitation_new(
                excitation_vector=excitation_vector,
                flag_matrix=flag_matrix,
                index_matrix=index_matrix,
                excitations=excitations,
                excitation_kwargs={"variable": 1.0},
                domain_indices=[
                    np.arange(0, n_rows, 1),
                    np.arange(0, n_cols, 1),
                ],
            )

            # plt.figure()
            # plt.pcolormesh(system_matrix, cmap=plt.cm.turbo)
            # plt.colorbar()
            # plt.show()

            # solve
            potentials = np.linalg.solve(system_matrix, excitation_vector)
            potentials = potentials.reshape(x_mesh.shape)

            # save results to file
            if dielectric is True:
                filename = f"fd_{step}_cell_{np.abs(limits[0])}db_{center_width}wb_w_dielectric.npz"
            else:
                filename = f"fd_{step}_cell_{np.abs(limits[0])}db_{center_width}wb.npz"
            with open(data_folder / filename, "wb") as fp:
                np.savez_compressed(
                    file=fp,
                    x_mesh=x_mesh,
                    y_mesh=y_mesh,
                    flag_matrix=flag_matrix,
                    system_matrix=system_matrix,
                    excitation_vector=excitation_vector,
                    potentials=potentials,
                )

            # plot solution
            if dielectric is True:
                title_label = f"{step} cells, {np.abs(limits[0])} d/b, {center_width} w/b w/ dielectric"
                filename = f"fd_{step}_cell_{np.abs(limits[0])}db_{center_width}wb_w_dielectric"
            else:
                title_label = (
                    f"{step} cells, {np.abs(limits[0])} d/b, {center_width} w/b"
                )
                filename = f"fd_{step}_cell_{np.abs(limits[0])}db_{center_width}wb"
            fig_potential = plt.figure()
            plt.pcolormesh(x_mesh, y_mesh, potentials, cmap=plt.cm.turbo)
            plt.colorbar()
            plt.title(f"Potentials, {title_label}")
            fig_potential.savefig(data_folder / f"{filename}_potential.png", dpi=100)
            fig_flags = plt.figure()
            plt.pcolormesh(
                x_mesh,
                y_mesh,
                flag_matrix,
                cmap=plt.cm.turbo,
                vmin=0,
                vmax=len(excitations),
            )
            plt.colorbar()
            plt.title(f"Flags, {title_label}")
            fig_flags.savefig(data_folder / f"{filename}_flags.png", dpi=100)
    plt.show()
