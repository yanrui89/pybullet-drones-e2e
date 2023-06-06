import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
import re
import sys

output_folder = os.path.join(sys.argv[1], sys.argv[2])
run_name = sys.argv[3]
results_folder = os.path.join(output_folder, run_name)
files = [f for f in glob.glob(f"{results_folder}/*.npy")]
sorted_files = sorted(files)

file = sorted_files[int(sys.argv[4])]
print(file)
data = np.load(file)

fig = plt.figure(figsize=(8, 6))
ax = plt.axes(projection='3d')

traj_x = data["states"][0, 0, :]
traj_y = data["states"][0, 1, :]
traj_z = data["states"][0, 2, :]

# print(traj_x[:20], traj_y[:20], traj_z[:20])

goal = data["goal"]

t = data["timestamps"][0]
ratio = t/t.max()

ax.scatter3D(*goal, c='r', marker='x')
p = ax.scatter(traj_x, traj_y, traj_z, s=1, c=t, cmap="viridis")
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)');
# ax.set_xlim([-5, 5])
# ax.set_ylim([-5, 5])
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.set_zlim([0, 4])
ax.view_init(34, -110)

fig.colorbar(p)
plt.show()
