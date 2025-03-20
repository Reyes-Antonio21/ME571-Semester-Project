import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob

def load_data(files):
    """Load .dat files into a list of numpy arrays."""
    data = []
    for file in sorted(files):
        frame = np.loadtxt(file)
        data.append(frame)
    return data

def animate_water_simulation(data):
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
    files = glob.glob("*.dat")
    if not files:
        print("No .dat files found!")
    else:
        data = load_data(files)
        animate_water_simulation(data)
