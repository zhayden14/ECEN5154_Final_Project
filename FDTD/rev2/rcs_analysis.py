"""
rcs_analysis.py
Zachary Hayden
ECEN5154, Fall 2022

Generates time-domain plots of fields near the position where the source current was applied
"""

#%% imports
import numpy as np
from matplotlib import pyplot as plt

#%% load data from file
# NOTE: this path is relatively benign, but it isn't included in the source repository
filename = r"D:\CU_GRAD\ECEN5154_CEM\Final_Project\repo\FDTD\rev2\outputs\2022-12-09_03-21-36_rcs.npz"
data = np.load(filename)

rcs_array = data["rcs_array"]
rcs_points_e = data["rcs_points_e"]
rcs_points_h = data["rcs_points_h"]
rcs_point_data = data["rcs_point_data"]

# %% plot "array" RCS data
rcs_array_normal = rcs_array[0:128] / np.max(rcs_array[0:128])
rcs_initial = rcs_array_normal[0:40].argmax()
rcs_reflected = rcs_array_normal[60:75].argmax() + 60
plt.figure()
plt.plot(rcs_array_normal)
plt.scatter(
    [rcs_initial, rcs_reflected],
    [rcs_array_normal[rcs_initial], rcs_array_normal[rcs_reflected]],
)
plt.annotate(
    f"t={rcs_initial}, Ex={rcs_array_normal[rcs_initial]:1.2f}",
    (rcs_initial + 4, rcs_array_normal[rcs_initial]),
)
plt.annotate(
    f"t={rcs_reflected}, Ex={rcs_array_normal[rcs_reflected]:1.2f}",
    (rcs_reflected + 2, rcs_array_normal[rcs_reflected] + 0.15),
)
plt.title("Sum of Ex at Y=40")
plt.xlabel("Time Sample")
plt.ylabel("Normalized Ex Sum")
plt.show()

# %% plot of point rcs
rcs_point_normal = rcs_point_data[0, 0:128] / np.max(rcs_point_data[0, 0:128])
rcs_initial = rcs_point_normal[0:40].argmax()
rcs_reflected = rcs_point_normal[60:75].argmax() + 60
plt.figure()
plt.plot(rcs_point_normal)
plt.scatter(
    [rcs_initial, rcs_reflected],
    [rcs_point_normal[rcs_initial], rcs_point_normal[rcs_reflected]],
)
plt.annotate(
    f"t={rcs_initial}, Ex={rcs_point_normal[rcs_initial]:1.2f}",
    (rcs_initial + 4, rcs_point_normal[rcs_initial]),
)
plt.annotate(
    f"t={rcs_reflected}, Ex={rcs_point_normal[rcs_reflected]:1.2f}",
    (rcs_reflected + 4, rcs_point_normal[rcs_reflected]),
)
plt.title("Ex at X=32, Y=40")
plt.xlabel("Time Sample")
plt.ylabel("Normalized Ex")
plt.show()

# %%
