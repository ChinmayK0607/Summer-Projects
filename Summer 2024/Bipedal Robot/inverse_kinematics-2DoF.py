import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider


a1 = 1
a2 = 1 

def inverse_kinematics(x, y):
    cos_th2 = (x**2 + y**2 - a1**2 - a2**2) / (2 * a1 * a2)
    if cos_th2 < -1 or cos_th2 > 1:
        return None, None
    th2 = np.arctan2(np.sqrt(1 - cos_th2**2), cos_th2)
    th1 = np.arctan2(y, x) - np.arctan2(a2 * np.sin(th2), a1 + a2 * np.cos(th2))
    return th1, th2

def manipulator(x, y):
    th1, th2 = inverse_kinematics(x, y)
    if th1 is None or th2 is None:
        print("The point is unreachable")
        return
    x1 = a1 * np.cos(th1)
    y1 = a1 * np.sin(th1)
    x2 = x1 + a2 * np.cos(th1 + th2)
    y2 = y1 + a2 * np.sin(th1 + th2)
    
    plt.figure()
    plt.plot([0, x1, x2], [0, y1, y2], 'ro-')
    plt.xlim(-(a1+a2), (a1+a2))
    plt.ylim(-(a1+a2), (a1+a2))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'th1: {np.degrees(th1):.2f}°, th2: {np.degrees(th2):.2f}°')
    plt.grid()
    plt.show()

x_end = FloatSlider(min=-(a1+a2), max=(a1+a2), step=0.001, value=a1, description='X:')
y_end = FloatSlider(min=-(a1+a2), max=(a1+a2), step=0.001, value=a2, description='Y:')

interact(manipulator, x=x_end, y=y_end)
