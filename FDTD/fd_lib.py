"""
Appendix A
Basic stencil and inding functions for FD solver

ECEN 5154 Homework 2
Zachary Hayden
"""

#%%
from os import system
from typing import List, Tuple
import itertools
import numpy as np

#%% convert node coordinates to index in array
def node_index_matrix(shape):
    """Creates an index matrix to converts numpy indexes to a row/column in the system matrix"""
    # total number of points is product of dimensions
    total = np.prod(np.array(shape))
    # generate 0 to total-1 indices, then reshape to match the requested dimensions
    index_vector = np.arange(start=0, stop=total, step=1)
    index_matrix = np.reshape(index_vector, shape)
    return index_matrix


def node_index(matrix, *args):
    """Uses the index matrix to get the requested index in the system matrix given the passed indices"""
    # Isn't numpy awesome? this just works
    return matrix[args]


#%% better stencil application

# TODO: test
# FIXME: rename this when complete
def apply_stencil_new(
    system_matrix: np.ndarray,
    flag_matrix: np.ndarray,
    index_matrix: np.ndarray,
    stencils: List[np.ndarray],  # assume each stencil has the correct dimensions
    stencil_extents: np.ndarray,
    domain_indices: List[np.ndarray],
) -> None:
    """Adds stencils to the system matrix. stencils may have arbitrary shape"""
    # NOTE stencils CANNOT be a ndarray. it must be a list of ndarrays to allow for arbitrary dimensions
    stencils = [np.array(idx) for idx in stencils]
    # force stencil_extents to be 3D: (# of stencils, # of dimensions, 2)
    stencil_extents = np.atleast_3d(stencil_extents)
    # TODO: the preceding should be handled at the level above. do we need these checks?
    # force each dimension to have an ndarray of indices
    domain_indices = [np.atleast_1d(idx) for idx in domain_indices]

    # I'd love to do this in numpy, but indexing is depent on the extents of each stencil.
    # I don't see a good way to do this without iterators
    apply_points = itertools.product(*tuple(domain_indices))
    for point in apply_points:
        # get requested stencil
        stencil_number = flag_matrix[point]
        stencil = stencils[stencil_number]
        # get the min/max relative coordinates of the stencil
        stencil_min = stencil_extents[stencil_number, :, 0]
        stencil_max = stencil_extents[stencil_number, :, 1]
        # get slices in index matrix for stencil of this shape
        index_slices = tuple(
            [
                slice(point[dim] + stencil_min[dim], point[dim] + stencil_max[dim] + 1)
                for dim in range(len(point))
            ]
        )
        # get indices in system matrix
        stencil_indices = index_matrix[index_slices]
        central_index = np.resize(index_matrix[point], stencil_indices.shape)
        # fill in system matrix
        system_matrix[central_index, stencil_indices] = stencil


# TODO: test
# FIXME: rename this when complete
def apply_excitation_new(
    excitation_vector: np.ndarray,
    flag_matrix: np.ndarray,
    index_matrix: np.ndarray,
    excitations: List[callable],
    excitation_kwargs: dict,
    domain_indices: List[np.ndarray],
    pre_calculate: bool = True,  # assumes most excitation vectors will be large relative to list of all excitations
) -> None:
    """Adds stencils to the system matrix. stencils may have arbitrary shape"""
    # NOTE: excitation callables take *args and **kwargs arguments, provided for each run

    # force each dimension to have an ndarray of indices
    domain_indices = [np.atleast_1d(idx) for idx in domain_indices]

    # pre-calculate all excitation values, to potentially save on execution time
    if pre_calculate is True:
        excitation_values = [
            excitation_func(**excitation_kwargs) for excitation_func in excitations
        ]

    # I'd love to do this in numpy indexing, but this way is a lot clearer
    apply_points = itertools.product(*tuple(domain_indices))
    for point in apply_points:
        # apply appropriate excitation
        if pre_calculate is False:
            # call the function during assignment
            excitation_vector[index_matrix[point]] = excitations[flag_matrix[point]](
                **excitation_kwargs
            )
        else:
            # used the cached result
            excitation_vector[index_matrix[point]] = excitation_values[
                flag_matrix[point]
            ]


#%% define generic stencils
def stencil_iface_dielectric(
    er_ne: float,
    er_nw: float,
    er_sw: float,
    er_se: float,
    h_e: float,
    h_n: float,
    h_w: float,
    h_s: float,
) -> Tuple[np.ndarray, float]:
    """Generic charge-free, central difference stencil"""
    # NOTE: epsilon relative and h (distance) defined in terms of cardinal directions
    # for now, first order central difference only
    # each term of phi_0 coefficient on its own line
    phi_0 = -1 * er_ne * h_n / (2 * h_e)
    phi_0 -= er_ne * h_e / (2 * h_n)
    phi_0 -= er_nw * h_w / (2 * h_n)
    phi_0 -= er_nw * h_n / (2 * h_w)
    phi_0 -= er_sw * h_s / (2 * h_w)
    phi_0 -= er_sw * h_w / (2 * h_s)
    phi_0 -= er_se * h_e / (2 * h_s)
    phi_0 -= er_se * h_s / (2 * h_e)
    # each other coefficient on its own line
    phi_e = (er_ne * h_n + er_se * h_s) / (2 * h_e)
    phi_n = (er_ne * h_e + er_nw * h_w) / (2 * h_n)
    phi_w = (er_nw * h_n + er_sw * h_s) / (2 * h_w)
    phi_s = (er_sw * h_w + er_se * h_e) / (2 * h_s)
    # define the stencil array
    stencil = np.array(
        [[0, phi_n, 0], [phi_w, phi_0, phi_e], [0, phi_s, 0]], dtype=float
    )
    return stencil, 0.0


#%%
def apply_stencil(
    system_matrix: np.ndarray,
    excitation_vector: np.ndarray,
    index_matrix: np.ndarray,
    stencils: np.ndarray,  # TODO: combine this and the next into a class?
    # NOTE: ALL stencils must have equal dimensions
    excitations: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> None:
    """Adds 3x3 stencils to the system matrix. stencils, excitations, rows and columns may be vectorized"""
    # NOTE: this only works with 3x3 stencils that are fully enclosed by the computational domain
    # force rows to ndarray with at least 1 dimension
    rows = np.atleast_1d(rows)
    cols = np.atleast_1d(cols)
    stencils = np.atleast_1d(stencils)
    excitations = np.atleast_1d(excitations)
    # get row and column indices for each element of each stencil
    # TODO: extend for multiple sizes/shapes?
    stencil_rows = np.linspace(rows - 1, rows + 1, 3, dtype=int, axis=-1)
    stencil_cols = np.linspace(cols - 1, cols + 1, 3, dtype=int, axis=-1)
    # expand stencil row and column indices into a grid
    row_grid, col_grid = np.meshgrid(stencil_rows, stencil_cols, indexing="ij")
    # get system matrix indices of all points in the grid
    stencil_indices = node_index(index_matrix, row_grid, col_grid)
    stencil_indices = stencil_indices.reshape((rows.size, 3, cols.size, 3))
    stencil_indices = np.moveaxis(stencil_indices, [0, 1, 2, 3], [0, 2, 1, 3])
    # get central index (for row/col of system matrix)
    # TODO: eventaully add shapes other than (3,3)
    central_indices = stencil_indices[:, :, 1, 1]
    # knead this into the same coordinates as stencil indices
    central_indices = np.resize(central_indices, (9, rows.size * cols.size))
    central_indices = central_indices.transpose()
    central_indices = central_indices.reshape(stencil_indices.shape)
    # apply to system matrix
    system_matrix[central_indices, stencil_indices] = stencils
    excitation_vector[central_indices[:, :, 0, 0]] = excitations
    # print(central_indices)
    # print(stencil_indices)


def apply_stencil_1x1(
    system_matrix: np.ndarray,
    excitation_vector: np.ndarray,
    index_matrix: np.ndarray,
    stencils: np.ndarray,  # TODO: combine this and the next into a class?
    # NOTE: ALL stencils must have equal dimensions
    excitations: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> None:
    """Applies 1x1 stencils to the system matrix. Typically used for PEC BC at the edge of the computational domain"""
    # force rows to ndarray with at least 1 dimension
    rows = np.atleast_1d(rows)
    cols = np.atleast_1d(cols)
    stencils = np.atleast_1d(stencils)
    excitations = np.atleast_1d(excitations)
    # get row and column indices for each element of each stencil
    # TODO: extend for multiple sizes/shapes?
    # stencil_rows = np.linspace(rows - 1, rows + 1, 3, dtype=int, axis=-1)
    # stencil_cols = np.linspace(cols - 1, cols + 1, 3, dtype=int, axis=-1)
    # expand stencil row and column indices into a grid
    row_grid, col_grid = np.meshgrid(rows, cols, indexing="ij")
    # get system matrix indices of all points in the grid
    stencil_indices = node_index(index_matrix, row_grid, col_grid)
    # apply to system matrix
    # TODO: the indexing of stencils is hacky and requires 3x3. how to extend?
    system_matrix[stencil_indices, stencil_indices] = stencils[:, :, 1, 1]
    excitation_vector[stencil_indices] = excitations
    # print(central_indices)
    # print(stencil_indices)


#%% test
if __name__ == "__main__":
    """test apply_stencil on a small system matrix to confirm correct indexing"""
    rcount = 8
    ccount = 8
    system_matrix = np.zeros((rcount * ccount, rcount * ccount), dtype=float)
    excitation_vector = np.zeros((rcount * ccount), dtype=float)
    index_matrix = node_index_matrix(rows=rcount, columns=ccount)

    # generate list of stencils
    stencils = [
        np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
        np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
        np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
    ]
    excitations = [
        0,
        0,
        1,
    ]

    apply_stencil(
        system_matrix,
        excitation_vector,
        index_matrix,
        stencils[0],
        0,
        rows=1,
        cols=1,
    )
    apply_stencil(
        system_matrix,
        excitation_vector,
        index_matrix,
        stencils[2],
        0,
        rows=[1, 2],
        cols=[2, 3],
    )
    apply_stencil(
        system_matrix,
        excitation_vector,
        index_matrix,
        stencils[0],
        0,
        rows=4,
        cols=[2, 3, 3],
    )
    apply_stencil(
        system_matrix,
        excitation_vector,
        index_matrix,
        stencils[2],
        0,
        rows=[1, 6],
        cols=6,
    )

    import matplotlib.pyplot as plt

    plt.figure()
    plt.pcolormesh(system_matrix, cmap=plt.cm.turbo)
    plt.colorbar()
    plt.show()
