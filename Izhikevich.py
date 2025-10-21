# from "Simple Model of Spiking Neurons" && "Analysis of Chaotic Resonance in Izhikevich Neuron Model"
# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1257420
# https://journals.plos.org/plosone/article/file?id=10.1371%2Fjournal.pone.0138919&type=printable

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

DELTA_T: float = 0.1  # Time step in ms

class IzhikevichNeuron:
    def __init__(self, a=0.2, b=2, c=-56, d=-16):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.v = c  # Membrane potential
        self.u = self.b * self.v  # Recovery variable
        self.dv = 0.0
        self.du = 0.0

    def step(self, I, dt=DELTA_T):
        for _ in range(1):
            if self.v >= 30:
                self.v = self.c
                self.u += self.d
            self.dv = (0.04 * self.v ** 2 + 5 * self.v + 140 - self.u + I) * dt
            self.du = (self.a * (self.b * self.v - self.u)) * dt
            self.v += self.dv
            self.u += self.du
        return self.u, self.v, self.du, self.dv

def simulate_neuron(neuron, I, duration=1000, dt=DELTA_T):
    time_steps = int(duration / dt)
    u_trace = np.zeros(time_steps)
    v_trace = np.zeros(time_steps)
    du_trace = np.zeros(time_steps)
    dv_trace = np.zeros(time_steps)
    for t in range(time_steps):
        u_trace[t], v_trace[t], du_trace[t], dv_trace[t] = neuron.step(I, dt)
    return u_trace, v_trace, du_trace, dv_trace

def plot_results(v_trace, dt=DELTA_T):
    time_axis = np.arange(len(v_trace)) * dt
    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, v_trace)
    plt.title('Izhikevich Neuron Simulation')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')
    plt.grid()
    plt.savefig('izhikevich_output.png', dpi=150, bbox_inches='tight')  # 保存
    print("Plot saved as 'izhikevich_output.png'")
    # plt.show()

def plot_uv(u_trace, v_trace, dt=DELTA_T):
    time_axis = np.arange(len(v_trace)) * dt
    plt.figure(figsize=(10, 5))
    plt.plot(v_trace, u_trace)
    plt.title('Izhikevich Neuron Simulation: u and v')
    plt.xlabel('Membrane Potential (v)')
    plt.ylabel('Recovery Variable (u)')
    plt.grid()
    plt.savefig('izhikevich_uv_output.png', dpi=150, bbox_inches='tight')  # 保存
    print("Plot saved as 'izhikevich_uv_output.png'")
    # plt.show()

# The def. of Nullclines is where dv/dt = 0 and du/dt = 0.
def plot_nullclines_and_vectors(neuron, I, v_range=(-80, 60), u_range=(-125, -80), dt=DELTA_T):
    v = np.linspace(v_range[0], v_range[1], 400)
    u = np.linspace(u_range[0], u_range[1], 400)
    V, U = np.meshgrid(v, u)

    dV = 0.04 * V**2 + 5 * V + 140 - U + I
    dU = neuron.a * (neuron.b * V - U)

    plt.figure(figsize=(10, 7))
    plt.streamplot(V, U, dV, dU, color='lightgray', density=1.5)

    # Nullclines
    u_nullcline = 0.04 * v**2 + 5 * v + 140 + I
    v_nullcline = neuron.b * v

    plt.plot(v, u_nullcline, 'r-', label='dv/dt=0 Nullcline')
    plt.plot(v, v_nullcline, 'b-', label='du/dt=0 Nullcline')

    plt.xlim(v_range)
    plt.ylim(u_range)
    plt.xlabel('Membrane Potential (v)')
    plt.ylabel('Recovery Variable (u)')
    plt.title('Nullclines and Vector Field of Izhikevich Neuron')
    plt.legend()
    plt.grid()
    plt.savefig('izhikevich_nullclines.png', dpi=150, bbox_inches='tight')  # 保存
    print("Plot saved as 'izhikevich_nullclines.png'")
    # plt.show()

if __name__ == "__main__":
    neuron = IzhikevichNeuron()
    I = -99  # Input current
    duration = 1000  # Simulation duration in ms
    dt = DELTA_T  # Time step in ms

    start_time = time.time()
    u_trace, v_trace, du_trace, dv_trace = simulate_neuron(neuron, I, duration, dt)
    end_time = time.time()

    print(f"Simulation completed in {end_time - start_time:.4f} seconds.")
    plot_results(v_trace, dt)
    plot_uv(u_trace, v_trace, dt)
    plot_nullclines_and_vectors(neuron, I)
    print(v_trace)
    print(u_trace)
