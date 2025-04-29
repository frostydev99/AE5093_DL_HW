import matplotlib.pyplot as plt
import numpy as np

def plot_disturbances(latitudes, longitudes, delta_B):
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(longitudes, latitudes, c=delta_B, cmap='plasma', s=50)
    plt.colorbar(sc, label='Disturbance Magnitude (µT)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Magnetic Disturbance Detection')
    plt.grid(True)
    plt.show()
