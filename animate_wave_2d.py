import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
import scipy.interpolate
from mpl_toolkits.mplot3d import Axes3D
import glob

# Function to load data from a given file
def load_data(file_path):
    data = np.loadtxt(file_path, delimiter="\t")
    x_cords = data[:, 0]
    y_cords = data[:, 1]
    water_height = data[:, 2]
    return x_cords, y_cords, water_height

# File pattern to match all .dat files (assumes they are named with time steps)
file_list = sorted(glob.glob("tc_2d_0.*.dat"))  # Sort to ensure correct time order

# Load first file to set up the plot
x_cords, y_cords, water_height = load_data(file_list[0])

nx = int(math.sqrt(len(x_cords)))

# Set up a regular grid of interpolation points
xi, yi = np.linspace(min(x_cords), max(x_cords), nx), np.linspace(min(y_cords), max(y_cords), nx)
xi, yi = np.meshgrid(xi, yi)

# Interpolate; there's also method='cubic' for 2-D data such as here
water_height_initial = scipy.interpolate.griddata((x_cords, y_cords), water_height, (xi, yi), method='linear')

# Set up the figure and 3D axis
fig = plt.figure()

# `ax` is a 3D-aware axis instance, because of the projection='3d' keyword argument to add_subplot
ax = fig.add_subplot(1, 1, 1, projection ='3d')
ax.plot_surface(xi, yi, water_height_initial, rstride = 1, cstride = 1, cmap = cm.coolwarm, linewidth = 0, alpha = 0.75)

# Set limits for x, y, & z axes
ax.set_xlim(min(x_cords), max(x_cords)) 
ax.set_ylim(min(y_cords), max(y_cords))
ax.set_zlim(0.75, 1.25)

# Set axis labels and title
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.set_zlabel("Water Height")
ax.set_title("Water Height Animation")
ax.view_init(elev = 30, azim = 220)  # Adjust viewing angle

# Update function for animation
def update(frame):
    x_cords, y_cords, water_height = load_data(file_list[frame])
    nx = int(math.sqrt(len(x_cords)))
    xn, yn = np.linspace(min(x_cords), max(x_cords), nx), np.linspace(min(y_cords), max(y_cords), nx)
    xn, yn = np.meshgrid(xn, yn)
    water_height_n = scipy.interpolate.griddata((x_cords, y_cords), water_height, (xn, yn), method='linear')
    ax.cla()
    ax.plot_surface(xn, yn, water_height_n, rstride = 1, cstride = 1, cmap = cm.coolwarm, linewidth = 0, alpha = 0.75)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Water Height")
    ax.set_title(f"Water Height at Time Step: {frame * 0.004:.3f} sec")
    ax.view_init(elev = 30, azim = 220)
    return ax,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames = len(file_list), interval = 100, blit = False)

# Show the animation
plt.show()