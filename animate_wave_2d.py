import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob

def load_data(files):
    """Load .dat files into a list of numpy arrays."""
    data = []
    for file in sorted(files):
        frame = pd.read_csv(file, delimiter="\t")
        data.append(frame)
    return data

def animate_Shallow_Water_Equations(data):
    """Create an animation from the simulation data."""
    fig, ax = plt.subplots()
    cax = ax.imshow(data[0], cmap='viridis', interpolation='nearest')
    fig.colorbar(cax)
    
    def update(frame):
        cax.set_array(data[frame])
        return [cax]
    
    ani = animation.FuncAnimation(fig, update, frames = len(data), interval=100, blit=False)
    plt.show()

if __name__ == "__main__":

    files = glob.glob("tc_2d_0.*.dat")
    
    data = load_data(files)

    animate_Shallow_Water_Equations(data)
