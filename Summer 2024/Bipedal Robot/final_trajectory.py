import numpy as np
import matplotlib.pyplot as plt
from math import atan2, cos, sin, sqrt, degrees

# Link lengths
a1, a2 = 7.3, 6


def inverse_kinematics(x, y):
    r = sqrt(x * 2 + y * 2)  # Fix calculation for r

    if r > (a1 + a2):
        raise ValueError("Target is not reachable")

    cos_theta2 = (r * 2 - a1 * 2 - a2 ** 2) / (2 * a1 * a2)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
    sin_theta2 = sqrt(1 - cos_theta2 ** 2)
    theta2 = atan2(sin_theta2, cos_theta2)

    theta1 = atan2(y, x) - atan2(a2 * sin(theta2), a1 + a2 * cos(theta2))

    x1 = a1 * cos(theta1)
    y1 = a1 * sin(theta1)

    x2 = x1 + a2 * cos(theta1 + theta2)
    y2 = y1 + a2 * sin(theta1 + theta2)

    return [0, x1, x2], [0, y1, y2], x2, y2, [theta1, theta2]


# Parameters for the half sinusoidal wave
amplitude = 3
num_points = 50

# Define the range for x-values (end effector position along the x-axis)
x_values_sin = np.linspace(-1, 4.5, num_points)

# Define the half-sinusoidal wave in the y-axis, shifted down by 10 units (to simulate the ground)
y_values_sin = -12.5 + amplitude * np.sin(
    np.pi * (x_values_sin - x_values_sin.min()) / (x_values_sin.max() - x_values_sin.min()))

# Retrace path: straight line along the x-axis at y = -10 from 5 to -1
x_values_retrace = np.linspace(4.5, -1, num_points)
y_values_retrace = np.full_like(x_values_retrace, -12.5)

# Combine the sinusoidal and retrace paths
x_values = np.concatenate([x_values_sin, x_values_retrace])
y_values = np.concatenate([y_values_sin, y_values_retrace])

# Create the figure and axis for the plot
fig, ax = plt.subplots()
ax.set_xlim(-18, 18)
ax.set_ylim(-18, 0)
ax.set_aspect('equal')  # Ensure equal scaling
ax.grid(True)

# Lists to store the traced path of the end effector
trace_x = []
trace_y = []

# List to store the angles
angles = []

# Plot the robot arm for each point in the trajectory
for i in range(len(x_values)):
    x = x_values[i]
    y = y_values[i]

    x_data, y_data, x_end, y_end, theta_values = inverse_kinematics(x, y)

    # Clear the previous plot
    ax.cla()
    ax.set_xlim(-18, 18)
    ax.set_ylim(-18, 0)
    ax.set_aspect('equal')
    ax.grid(True)

    # Plot the arm
    ax.plot(x_data, y_data, 'o-', lw=2, color='blue')

    # Update trace data
    trace_x.append(x_end)
    trace_y.append(y_end)

    # Plot the trace path of the end effector
    ax.plot(trace_x, trace_y, 'r--', lw=1)

    # Convert angles to degrees and annotate
    theta1_deg = degrees(theta_values[0])
    theta2_deg = degrees(theta_values[1])

    ax.text(-17, -1, f'θ1: {theta1_deg:.2f}°', fontsize=12, color='blue')
    ax.text(-17, -2, f'θ2: {theta2_deg:.2f}°', fontsize=12, color='green')

    # Save the angles
    angles.append([theta1_deg, theta2_deg])

    # Pause to simulate animation effect (optional: adjust the pause time to change speed)
    plt.pause(0.1)

# Save angles to a text file
with open('angles_sinusoidal_retrace.txt', 'w') as f:
    for angle in angles:
        f.write(f"{angle[0]:.6f}, {angle[1]:.6f},\n")

print("Angles saved to angles_sinusoidal_retrace.txt")

# Show the final plot
plt.show()
