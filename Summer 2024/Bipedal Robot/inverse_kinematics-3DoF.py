import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider

a1 = 1
a2 = 1
a3 = 1

def inverse_kinematics(x, y, z):
    cos_th2 = (x**2 + y**2 + z**2 - a1**2 - a2**2 - a3**2) / (2 * a1 * a2)
    if cos_th2 < -1 or cos_th2 > 1:
        return None, None, None
    th2 = np.arctan2(np.sqrt(1 - cos_th2**2), cos_th2)
    th1 = np.arctan2(y, x) - np.arctan2(a2 * np.sin(th2), a1 + a2 * np.cos(th2))
    th3 = np.arctan2(z, np.sqrt(x**2+y**2))
    return th1, th2, th3

def manipulator(x, y, z):
    th1, th2, th3 = inverse_kinematics(x, y, z)
    if th1 is None or th2 is None or th3 is None:
        print("The point is unreachable")
        return
    x1 = a1 * np.cos(th1)
    y1 = a1 * np.sin(th1)
    z1 = 0
    
    x2 = x1 + a2 * np.cos(th1 + th2)
    y2 = y1 + a2 * np.sin(th1 + th2)
    z2 = z1 + a3 * np.sin(th3)
    
    x3 = x2
    y3 = y2
    z3 = z2 + a3 * np.cos(th3)
    
    plt.figure()
    plot = plt.axes(projection='3d')
    plot.plot([0, x1, x2, x3], [0, y1, y2, y3], [0, z1, z2, z3], 'bo-', markersize=10)
    plot.set_xlim(-(a1+a2+a3), (a1+a2+a3))
    plot.set_ylim(-(a1+a2+a3), (a1+a2+a3))
    plot.set_zlim(-(a1+a2+a3), (a1+a2+a3))
    plot.set_xlabel('X')
    plot.set_ylabel('Y')
    plot.set_zlabel('Z')
    plot.set_title(f'th1: {np.degrees(th1):.2f}°, th2: {np.degrees(th2):.2f}°, th3: {np.degrees(th3):.2f}°')
    plot.grid()
    plt.show()

x_end = FloatSlider(min=-(a1+a2+a3), max=(a1+a2+a3), step=0.001, value=a1, description='X:')
y_end = FloatSlider(min=-(a1+a2+a3), max=(a1+a2+a3), step=0.001, value=a2, description='Y:')
z_end = FloatSlider(min=-(a1+a2+a3), max=(a1+a2+a3), step=0.001, value=a3, description='Z:')

interact(manipulator, x=x_end, y=y_end, z=z_end)

