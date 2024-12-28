# BIPEDAL ROBOT
![WhatsApp Image 2024-11-15 at 4 49 05 PM](https://github.com/user-attachments/assets/a5e59df9-59af-4c63-8f5e-0d8ce6466e73)
## CONTENTS

- [Objective](#objective)
- [Tasks](#tasks)
  - [Task 1 : Cruise Controller](#task-1-cruise-controller)
  - [Task 2 : Forward & Inverse Kinematics Solutions](#task-2-forward--inverse-kinematics)
  - [Task 3 : Trajectory Tracing](#task-3-trajectory-tracing)
- [Project Implementation](#project-implementation)
- [Design](#design)
- [Electronic Components](#electronic-components)
- [Final Implementation](#final-implementation)
- [References](#references)

## OBJECTIVE
This project aims to develop a bipedal robot, a two-legged robotic system designed to replicate human gait dynamics. Here, the robot’s movement is controlled by precise limb positioning through inverse kinematics, while stability is maintained by managing the center of mass during walking. This repository provides an overview of the fundamental concepts, codes and model for simulating and executing stable bipedal locomotion, with an emphasis on kinematics and balance control.

## TASKS

### Task 1: Cruise Controller 
#### Aim
To develop a Cruise Controller using Python and Matlab(simulink) for maintaining desired set point velocity of a car.

**Simulink circuit design:**

![Screenshot 2024-11-07 235542](https://github.com/user-attachments/assets/b502fb53-0f23-4376-9cfc-54237ecd8016)

#### Result

The PID controller was successfully tuned to achieve:
- A rise time of approximately 10 seconds.
- A maximum overshoot of less than 5%.

**Python result:**
![SyKU5ti5C](https://github.com/user-attachments/assets/12fbe362-feef-4fc4-8daf-3996e7bcd4f1)

**Simulink result:**
![Screenshot 2024-11-07 235517](https://github.com/user-attachments/assets/a90c50b4-ed2a-40e9-8107-4ad6e709757c)

---

### Mechanics and Control of Robotic Manipulators

In bipedal robots, kinematics plays a crucial role in defining and controlling movement. Kinematics is divided into forward kinematics and inverse kinematics, both of which are essential for achieving precise and coordinated motion in robotic manipulators.

- **Forward Kinematics:** Forward kinematics involves calculating the robot's end-effector position based on given joint angles. In a bipedal robot, forward kinematics is used to determine the exact position of each leg segment when joint angles are set, enabling accurate placement of the feet during walking.

- **Inverse Kinematics:** Inverse kinematics is the reverse process, where the desired position of the end-effector (such as the foot) is specified, and the necessary joint angles are calculated to achieve that position. This is particularly important in bipedal robots for defining precise leg trajectories that maintain stability and allow for smooth, balanced walking motions.

### Task 2: Forward & Inverse Kinematics 
#### Aim
To find FK/IK solutions of Manipulator arm in XY Plane in Python Using matplotlib 
#### Forward Kinematics Result:
- Successfully defined the DH parameters and transformation matrices.
- The forward kinematics function accurately calculates the end-effector position for the specified assembly.

![WhatsApp Video 2024-11-08 at 2 56 41 PM](https://github.com/user-attachments/assets/bbcc3a39-8b33-4206-93d2-557aeead6b1e)

![WhatsApp Video 2024-11-08 at 5 23 16 PM](https://github.com/user-attachments/assets/70865eed-6e1d-4e9a-b3c7-0176f15c5572)

#### Inverse Kinematics Result:
- Successfully defined the DH parameters and transformation matrices.
- The inverse kinematics function accurately calculates the necessary joint angles for the specified end-effector position.

![WhatsApp Video 2024-11-08 at 3 51 17 PM](https://github.com/user-attachments/assets/41cdd9e7-82ab-4d8d-9c85-c435e8f71b97)

![WhatsApp Video 2024-11-08 at 5 23 24 PM](https://github.com/user-attachments/assets/6d017b42-6ecf-463b-bfee-0a550df742df)

### Task 3: Trajectory Tracing 
#### Aim 
To simulate 'a manipulator arm tracing a trajectory' in the XYZ plane using Python and Matplotlib.
#### Result
- Successfully simulated the arm manipulator in the XYZ plane using the chosen method.
- The IK parameters were accurately determined, and the manipulator followed the intended trajectory in the simulation.

![By8LkHSWJx](https://github.com/user-attachments/assets/5d0eb689-a231-469b-8235-3af9d5096b50)

![BJlDkrSWkg](https://github.com/user-attachments/assets/41201e08-5798-4b3e-a81c-6a34fc4ff971)

---

## PROJECT IMPLEMENTATION
- We decided to make 2dof manipulator as Bipedal Robot legs.
- Formed the inverse kinematics equations and solved them on the Sinusoidal trajectory followed by retracment of Straight Line trajectory backwards.
- Here is the python pseudocode for finding ik solutions:
```python
def inverse_kinematics(x, y):
    r = sqrt(x**2 + y**2)

    if r > (a1 + a2):
        raise ValueError("Target is not reachable")

    cos_theta2 = (r**2 - a1**2 - a2**2) / (2 * a1 * a2)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
    sin_theta2 = sqrt(1 - cos_theta2**2)
    theta2 = atan2(sin_theta2, cos_theta2)

    theta1 = atan2(y, x) - atan2(a2 * sin(theta2), a1 + a2 * cos(theta2))

```

#### Result
- Successfully simulated the 2DoF in XY Plane according to the required trajectory.

![rJNpf4H-yx](https://github.com/user-attachments/assets/9a32321c-6504-4178-abde-647eb72cf70a)

---


## DESIGN
- We used Autodesk Fusion 360 Software to create the CAD model for our 2DoF robot.
![Untitledvideo-MadewithClipchamp-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/16161beb-5bbe-4efb-8737-0b6dab87f72a)


---

## ELECTRONIC COMPONENTS
We used 3 main components in our project.

| **Component** | **Application** |
|---------------|------------------------------------------|
| Arduino UNO   | Used Arduino UNO to connect all electronic equipments. |  
| Servo motors  | Placed 6 servos with 3 servos on each leg: one in hip, one in knee and one in foot (to balance the COM while the robot is walking). |
| PCA9685       | Used PCA9685 (with a 5V power supply) to connect and power multiple servo motors easily. |
---

## FINAL IMPLEMENTATION

![Untitleddesign-ezgif com-video-speed](https://github.com/user-attachments/assets/e2e18482-1649-4838-ac25-0335463a5be7)

## REFERENCES
- https://www.youtube.com/playlist?list=PLyqSpQzTE6M-tWPjnJjFo9sHGWxgCnGrh
- https://youtube.com/playlist?list=PLn8PRpmsu08pQBgjxYFXSsODEF3Jqmm-y&si=w728cftgoCuM-PjS
- https://youtube.com/playlist?list=PLUMWjy5jgHK1NC52DXXrriwihVrYZKqjk&si=36dh-z4ip2kWO_XO
