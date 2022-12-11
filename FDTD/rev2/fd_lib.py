"""
fd_lib.py
Zachary Hayden
ECEN5154, Fall 2022

Basic functionality for indexing into a system matrix 
and populating a system matrix using stencils
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
def apply_stencil(
    system_matrix: np.ndarray,
    flag_matrix: np.ndarray,
    index_matrix_rows: np.ndarray,
    index_matrix_columns: np.ndarray,
    stencils: List[np.ndarray],
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
        stencil_indices = index_matrix_columns[index_slices]
        central_index = np.resize(index_matrix_rows[point], stencil_indices.shape)
        # fill in system matrix
        system_matrix[central_index, stencil_indices] = stencil


# TODO: test
# TODO: not needed for FDTD implementation?
# # FIXME: rename this when complete
# def apply_excitation_new(
#     excitation_vector: np.ndarray,
#     flag_matrix: np.ndarray,
#     index_matrix: np.ndarray,
#     excitations: List[callable],
#     excitation_kwargs: dict,
#     domain_indices: List[np.ndarray],
#     pre_calculate: bool = True,  # assumes most excitation vectors will be large relative to list of all excitations
# ) -> None:
#     """Adds stencils to the system matrix. stencils may have arbitrary shape"""
#     # NOTE: excitation callables take *args and **kwargs arguments, provided for each run

#     # force each dimension to have an ndarray of indices
#     domain_indices = [np.atleast_1d(idx) for idx in domain_indices]

#     # pre-calculate all excitation values, to potentially save on execution time
#     # NOTE: this is incompatible with spatially-varying excitations. I may need to remove this in the future
#     if pre_calculate is True:
#         excitation_values = [
#             excitation_func(**excitation_kwargs) for excitation_func in excitations
#         ]

#     # I'd love to do this in numpy indexing, but this way is a lot clearer
#     apply_points = itertools.product(*tuple(domain_indices))
#     for point in apply_points:
#         # apply appropriate excitation
#         if pre_calculate is False:
#             # call the function during assignment
#             excitation_vector[index_matrix[point]] = excitations[flag_matrix[point]](
#                 **excitation_kwargs
#             )
#         else:
#             # used the cached result
#             excitation_vector[index_matrix[point]] = excitation_values[
#                 flag_matrix[point]
#             ]
