import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import sys
def printf(Vx,Vy,Wz):
    print("Velocity of bot in X-axis(in units/min)=",Vx)
    print("Velocity of bot in Y-axis(in units/min)=",Vy)
    print("Angular Speed of bot(in RPM)=",Wz)

u1=int(input("Enter the RPM of 1st wheel:- "))
u2=int(input("Enter the RPM of 2nd wheel:- "))
u3=int(input("Enter the RPM of 3rd wheel:- "))
u4=int(input("Enter the RPM of 4th wheel:- "))

fig,ax = plt.subplots()
l,w,r=4,2,1

x1=x4=l
x2=x3=-l
y1=y2=w
y3=y4=-w
L=np.array([x1,x2,x3,x4,x1])
B=np.array([y1,y2,y3,y4,y1])
if r * u1 == r * u2 == r * u3 == r * u4:
    x = np.linspace(0, 1, 100) * (r*u1)
    y = np.linspace(0, 0, 100)
    Vx=r*u1*2*3.14
    Vy,Wz=0,0
    printf(Vx,Vy,Wz)
elif -r * u1 == r * u2 == -r * u3 == r * u4:
    x = np.linspace(0, 0, 100)
    y = np.linspace(0, 1, 100) * (r*u1)
    Vy=r*u1*2*3.14
    Vx,Wz=0,0
    printf(Vx,Vy,Wz)
elif -u1*r/(l+w) == u2*r/(l+w) == u3*r/(l+w) == -u4*r/(l+w):
    a=(w*w+l*l)**0.5
    Wz=-u1*r/(l+w)
    fhi=np.arccos((a*a-2*l*l)/(a*a))
    alpha=np.arccos((a*a-2*w*w)/(a*a))
    theta=np.linspace(np.arccos(l/a), 4*3.14+np.arccos(l/a), 100)*np.sign(u1)
    x = np.linspace(0, 0, 100)
    y = np.linspace(0, 0, 100)
    Vx,Vy=0,0
    printf(Vx,Vy,Wz)
elif u1==u3==0 and u2==u4:
    x = np.linspace(0, 1, 100)*(r*u2)/2
    y = np.linspace(0, 1, 100)*(r*u2)/2
    Wz=0
    Vx=Vy=(r*u2)*2*3.14/2
    printf(Vx,Vy,Wz)
elif u2==u4==0 and u1==u3:
    x = np.linspace(0, 1, 100)*(r*u1)/2
    y = np.linspace(0, -1, 100)*(r*u1)/2
    Wz=0
    Vx=Vy=(r*u1)*2*3.14/2
    printf(Vx,Vy,Wz)
elif u1==u2==u3==u4==0:
    print("Bot is in rest")
    sys.exit(0)
else:
    print("NO uniform motion")
    sys.exit(0)
ax.set(xlim=[-50, 50], ylim=[-50, 50])
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
plt.title("Omnidirectional bot's trajectory")
plt.grid(True)
square,= ax.plot(L, B)
def update(frame):
    if(Wz==-u1*r/(l+w)==u2*r/(l+w)):
        x1,x2=a*np.cos(theta[frame]),a*np.cos(theta[frame]+fhi)
        x3,x4=a*np.cos(theta[frame]+fhi+alpha),a*np.cos(theta[frame]+2*fhi+alpha)
        y1,y2=a*np.sin(theta[frame]),a*np.sin(theta[frame]+fhi)
        y3,y4=a*np.sin(theta[frame]+fhi+alpha),a*np.sin(theta[frame]+2*fhi+alpha)
    else:
        x1=x4=l
        x2=x3=-l
        y1=y2=w
        y3=y4=-w
    L=np.array([x1,x2,x3,x4,x1])+x[frame]
    B=np.array([y1,y2,y3,y4,y1])+y[frame]
    square.set_data(L, B)
    return square
ani = animation.FuncAnimation(fig=fig, func=update, frames=100, interval=1,repeat=True)
plt.show()
