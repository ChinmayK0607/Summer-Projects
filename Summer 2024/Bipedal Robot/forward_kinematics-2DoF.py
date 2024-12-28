import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import FloatSlider, interactive

def dh_transformation(alpha, a, theta, d):
    TDH = np.array([[np.cos(theta), -np.sin(theta), 0, a],
                    [np.sin(theta)*np.cos(alpha), np.cos(theta)*np.cos(alpha), -np.sin(alpha), -d*np.sin(alpha)],
                    [np.sin(theta)*np.sin(alpha), np.cos(theta)*np.sin(alpha), np.cos(alpha), d*np.cos(alpha)],
                    [0, 0, 0, 1]])
    return TDH

def forward_kinematics(th1, th2, a1, a2):
    th1 = np.deg2rad(th1)
    th2 = np.deg2rad(th2)
    DH_TABLE = [ [0, 0, th1, 0],
                 [0, a1, th2, 0],
                 [0, a2, 0, 0]   ]
       
    A = []
    for x in DH_TABLE:
        alpha, a, theta, d = x
        A.append(dh_transformation(alpha, a, theta, d))

    T = np.eye(4)
    positions = [(0, 0)]
    for i in range(len(A)):
        T = np.dot(T, A[i])
        tm = T[:2, 3]  
        positions.append((tm[0], tm[1]))
    return positions

def manipulator(th1, th2):
    a1 = 1  
    a2 = 1  
    positions = forward_kinematics(th1, th2, a1, a2)
    x, y = zip(*positions)
    plt.figure()
    plt.plot(x, y, 'bo-', markersize=10)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('2DoF Manipulator')
    plt.grid(True)
    x_end, y_end = positions[-1]
    plt.text(x_end, y_end, f'({x_end:.2f}, {y_end:.2f})', fontsize=10, ha='left')
    plt.show()

th1_slider = FloatSlider(min=-180, max=180, step=0.5, value=0, description='Theta1')
th2_slider = FloatSlider(min=-90, max=90, step=0.5, value=0, description='Theta2')

plot = interactive(manipulator, th1=th1_slider, th2=th2_slider)
display(plot)
