import matplotlib.pyplot as plt
import numpy as np

csv_PATH = 'izhikevich_ISI_chaos.csv'

def plot_return_map(csv_PATH):
    ISI = np.loadtxt(csv_PATH, delimiter=',') # Load ISI data from CSV
    ISI_n = ISI[:-1]
    ISI_n1 = ISI[1:]
    plt.figure(figsize=(8, 8))
    plt.scatter(ISI_n, ISI_n1, s=1, color='blue')
    plt.title('Return Map of Inter-Spike Intervals (ISI)')
    plt.xlabel('ISI_n (ms)')
    plt.ylabel('ISI_n+1 (ms)')
    plt.grid()
    plt.axis('equal')
    plt.savefig('izhikevich_ISI_return_map.png', dpi=150, bbox_inches='tight')
    print("Return map plot saved as 'izhikevich_ISI_return_map.png'")
    # plt.show()
