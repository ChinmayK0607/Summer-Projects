import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import FloatSlider, interactive  # Import FloatSlider and interactive

def dh_transformation(alpha, a, theta, d):
    TDH = np.array([[np.cos(theta), -np.sin(theta), 0, a],
                    [np.sin(theta)*np.cos(alpha), np.cos(theta)*np.cos(alpha), -np.sin(alpha), -d*np.sin(alpha)],
                    [np.sin(theta)*np.sin(alpha), np.cos(theta)*np.sin(alpha), np.cos(alpha), d*np.cos(alpha)],
                    [0, 0, 0, 1]])
    return TDH

def forward_kinematics(th1, th2, th3, a1, a2, a3):
    th1 = np.deg2rad(th1)
    th2 = np.deg2rad(th2)
    th3 = np.deg2rad(th3)
    DH_TABLE = [ [0, 0, th1, 0],  
                 [-np.pi/2, a1, th2, 0], 
                 [np.pi/2, a2, th3, 0],  
                 [0, a3, 0, 0]]

    A = []
    for x in DH_TABLE:
        alpha, a, theta, d = x
        A.append(dh_transformation(alpha, a, theta, d))

    T = np.eye(4)
    positions = [(0, 0, 0)]
    for i in range(len(A)):
        T = np.dot(T, A[i])
        tm = T[:3, 3]  
        positions.append((tm[0], tm[1], tm[2]))
    return positions

def manipulator(th1, th2, th3):
    a1 = 1  
    a2 = 1 
    a3 = 1
    positions = forward_kinematics(th1, th2, th3, a1, a2, a3)
    x, y, z = zip(*positions)
    plt.figure()
    plot = plt.axes(projection='3d')
    plot.plot(x, y, z, 'bo-', markersize=10)
    plot.set_xlim(-3, 3)
    plot.set_ylim(-3, 3)
    plot.set_zlim(-3, 3)
    plot.set_xlabel('X-axis')
    plot.set_ylabel('Y-axis')
    plot.set_zlabel('Z-axis')
    plot.set_title('3DoF Manipulator')
    plot.grid(True)
    x_end, y_end, z_end = positions[-1]
    plot.text(x_end, y_end, z_end, f'({x_end:.2f}, {y_end:.2f}, {z_end:.2f})', fontsize=10, ha='left')
    plt.show()

# Define sliders for joint angles
th1_slider = FloatSlider(min=-180, max=180, step=1, value=0, description='Theta1')
th2_slider = FloatSlider(min=-90, max=90, step=1, value=0, description='Theta2')
th3_slider = FloatSlider(min=-90, max=90, step=1, value=0, description='Theta3')

# Interactive plot
plot = interactive(manipulator, th1=th1_slider, th2=th2_slider, th3=th3_slider)
display(plot)
