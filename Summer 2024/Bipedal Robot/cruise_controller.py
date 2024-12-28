import numpy as np
import matplotlib.pyplot as plt
# Constants
mass = 1000 # mass of the car in kg
drag_coefficient = 50 # drag coefficient in Ns/m
# PID controller parameters tuned by Hit and Trial
Kp = 270
Ki = 15
Kd = 100
# Set-point velocity
v_set = 20 # desired velocity in m/s
# Simulation parameters
dt = 0.1 # time step in seconds
simulation_time = 50 # total simulation time in seconds
time = np.arange(0, simulation_time, dt)
# Initialize variables
v = np.zeros_like(time) # velocity
u = np.zeros_like(time) # input force
e = np.zeros_like(time) # error
integral = 0
previous_error = 0
# Simulation loop
for i in range(1, len(time)):
# Calculate the error
e[i] = v_set - v[i-1]
# Calculate the integral and derivative of the error
integral += e[i] * dt
derivative = (e[i] - previous_error) / dt
# Calculate the input force using PID control
u[i] = Kp * e[i] + Ki * integral + Kd * derivative
# Update the velocity using the dynamical model
v[i] = v[i-1] + (u[i] - drag_coefficient * v[i-1]) * dt / mass
# Update the previous error
previous_error = e[i]
# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(time, v, label='Velocity (m/s)')
plt.axhline(v_set, color='r', linestyle='--', label='Set-point')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Cruise Control System Response')
plt.legend()
plt.grid(True)
plt.show()
# Performance metrics
rise_time = next(t for t, velocity in zip(time, v) if velocity >= 0.9 * v_set)
max_overshoot = (max(v) - v_set) / v_set * 100
print(f"Rise Time: {rise_time:.2f} seconds")
print(f"Maximum Overshoot: {max_overshoot:.2f}%")
