FDTD library and code for Final Project of ECEN5154, Fall 2022

Source code within this folder was authored by Zachary Hayden, unless otherwise specified

source files are available at: https://github.com/zhayden14/ECEN5154_Final_Project/tree/main/FDTD/rev2

Required Dependencies:
----------------------
- Numpy
- Matplotlib

Recommended Development packages:
---------------------------------
- pipenv
- ipykernal

Index
-----
- fd_lib.py: Basic functionality for indexing into a system matrix and populating a system matrix using stencils
- fd_stencils.py: Functions to generate stencils. points_to_stencil creates a stencil from a list of points and values.
    The other functions define a standard set of stencils.
- fdtd_1d_scratchpad.py: 1D simulation to test stencils and PML for TFSF use
- fdtd_2d_scratchpad.py: 2D simulation for testing/development of 2D wave propagation
- fdtd_scattering.py: Main 2D simulation for scattering off an infinite PEC cylinder
- fdtd_solver.py: FDTD update function. Calculates the next E and H field values 
    and applies sources and PML boundary condition.
- rcs_analysis.py: Generate time-domain plots of fields near the position where the source current was applied
- tfsf.py: Unfinished total-field/scattered-field boundary condition implementation 
- unsplit_pml.py: Perfectly matched layer implementation