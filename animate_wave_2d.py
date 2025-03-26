import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
import scipy.interpolate
import glob

# Function to load data from a given file
def load_data(file_path):
    data = np.loadtxt(file_path, delimiter="\t")
    x_cords = data[:, 0]
    y_cords = data[:, 1]
    water_height = data[:, 2]
    return x_cords, y_cords, water_height

# File pattern to match all .dat files (assumes they are named with time steps)
file_list = sorted(glob.glob("tc2d_*.dat"))  # Sort to ensure correct time order

# Load first file to set up the plot
x_cords, y_cords, water_height = load_data(file_list[0])

nx = int(math.sqrt(len(x_cords)))

# Set up a regular grid of interpolation points
xi, yi = np.linspace(min(x_cords), max(x_cords), nx), np.linspace(min(y_cords), max(y_cords), nx)
xi, yi = np.meshgrid(xi, yi)

# Interpolate initial water height data
water_height_initial = scipy.interpolate.griddata((x_cords, y_cords), water_height, (xi, yi), method='linear')

# Set up the figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')

# Set limits for x, y, & z axes
ax.set_xlim(min(x_cords), max(x_cords)) 
ax.set_ylim(min(y_cords), max(y_cords))
ax.set_zlim(0.75, 1.25)

# Set axis labels and title
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.set_zlabel("Water Height")
ax.set_title("Water Height Animation")

# Update function for animation
def update(frame):
    x_cords, y_cords, water_height = load_data(file_list[frame])
    water_height_n = scipy.interpolate.griddata((x_cords, y_cords), water_height, (xi, yi), method='linear')
    
    ax.cla()  # Clear previous frame
    ax.plot_surface(xi, yi, water_height_n, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, alpha=0.75)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Water Height")
    ax.set_zlim(0.75, 1.25)
    ax.set_title(f"Water Height at Time Step: {frame * 0.004:.3f} sec")
    
    return ax,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(file_list), interval=4, blit=False)  # interval=4ms (~250 FPS)

# Save the animation as an MP4 video
mp4_filename = "water_height_animation.mp4"
writer = animation.FFMpegWriter(fps=250, bitrate=7500, extra_args=["-r", "250"])
ani.save(mp4_filename, writer=writer)

print(f"Animation saved as {mp4_filename}")

# Show the animation
plt.show()