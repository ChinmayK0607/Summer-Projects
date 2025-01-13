import matplotlib.pyplot as plt
import numpy as np
from math import sin, cos, atan2, sqrt, acos, pi

def ik(x, y, z, ax):
    a1 = 1
    a2 = 0.5
    a3 = 0.5

    # Calculate th1, th2, and th3 angles
    if x == 0:
        th1 = 0
    else:
        th1 = atan2(y, x)

    r1 = sqrt(x * x + y * y)
    r2 = z - a1
    ph2 = atan2(r2, r1)
    r3 = sqrt(r1 * r1 + r2 * r2)

    cos_angle1 = ((a3 * 2 - a2 * 2 - r3 ** 2) / (-2 * a2 * r3))
    cos_angle2 = ((r3 * 2 - a2 * 2 - a3 ** 2) / (-2 * a2 * a3))

    # keeping values between 1 and -1
    cos_angle1 = max(-1.0, min(1.0, cos_angle1))
    cos_angle2 = max(-1.0, min(1.0, cos_angle2))

    ph1 = acos(cos_angle1)
    th2 = ph2 - ph1
    ph3 = acos(cos_angle2)
    th3 = pi - ph3

        # Clear previous plot
    ax.cla()

    ax.plot([0, 0, a2 * cos(th1), x], [0, 0, a2 * sin(th1), y], [0, a1, a2 * sin(th2), z], 'b')
    ax.plot([0, 0, a2 * cos(th1), x], [0, 0, a2 * sin(th1), y], [0, a1, a2 * sin(th2), z], 'ro')

    x1 = np.linspace(0, 1, 100)
    z1 = 1
    y1 = np.ones_like(x1)
    ax.plot(x1, y1, z1, 'r--')

    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.set_title('3Dof Robotic Manipulator Straight Line Trajectory')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    plt.draw()
    plt.pause(0.1)

    return

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x2 = np.linspace(0, 1, 100)
frequency = 1
z2=1
for i in range(len(x2)):
    ik(x2[i], 1, z2, ax)

plt.ioff()
plt.show()
