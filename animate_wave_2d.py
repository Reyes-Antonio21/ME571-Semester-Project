import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
import scipy.interpolate
from mpl_toolkits.mplot3d import Axes3D
import glob

# File pattern to match all .dat files (assumes they are named with time steps)
file_list = sorted(glob.glob("tc_2d_0.*.dat"))  # Sort to ensure correct time order

# Function to load data from a given file
def load_data(file_path):
    data = np.loadtxt(file_path, delimiter="\t")
    x_coords = data[:, 0]
    y_coords = data[:, 1]
    water_height = data[:, 2]
    return x_coords, y_coords, water_height


# Load first file to set up the plot
x_coords, y_coords, water_height = load_data(file_list[0])

nx = int(math.sqrt(len(x_coords)))

# Set up a regular grid of interpolation points
xi, yi = np.linspace(min(x_coords), max(x_coords), nx), np.linspace(min(y_coords), max(y_coords), nx)
xi, yi = np.meshgrid(xi, yi)

# Set up the figure and 3D axis
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x_coords, y_coords, water_height, c=water_height, cmap="viridis", marker="o")
cbar = plt.colorbar(sc, label="Water Height")
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.set_zlabel("Water Height")
ax.set_title("Water Height Animation (3D)")
ax.view_init(elev=30, azim=220)  # Adjust viewing angle

# Update function for animation
def update(frame):
    x_coords, y_coords, water_height = load_data(file_list[frame])
    ax.cla()
    ax.scatter(x_coords, y_coords, water_height, c=water_height, cmap="viridis", marker="o")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Water Height")
    ax.set_title(f"Water Height at Time Step: {frame * 0.004:.3f} sec")
    ax.view_init(elev=30, azim=220)
    return ax,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(file_list), interval=100, blit=False)

# Show the animation
plt.show()