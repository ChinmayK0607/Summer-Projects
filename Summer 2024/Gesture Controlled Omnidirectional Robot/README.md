# Gesture-Controlled-Omnidirectional-Robot
#### This is the official repository for the omni directional hand gesture robot project at IvLabs, VNIT Nagpur.

# Objectives
##### The objectives of the summer internship were to create a omni directional robot which move in all 8 direction using transmitter receiver with hand gesture.

# Methodology
## Software Implementation:
### 1. Connection of MPU6050 with ESP32
#### Connected an MPU6050 to an ESP32 as the following connection. The MPU6050 is an inertial measurement unit(IMU) that is a combination of gyroscope and accelerometer in all 6 dimension, which can detect the orientation, motion, and acceleration of the hand and give its value of each change to ESP32.
<img src="https://hackmd.io/_uploads/SkmLKjrP0.png" width="300" height="300"/>

### 2. Simulation result of MPU6050
#### MPU6050 track each movement and send its value to ESP32 as shown in the result shown below.
<img src="https://github.com/user-attachments/assets/5237a5b0-dc07-4db9-a93a-7b81af47e2ab">

### 3. Motion of Omnidirectional robot

#### (i) Forward motion &emsp;&emsp;&emsp;&emsp; (ii) Backward motion &emsp;&emsp;&emsp;&emsp; (iii) Leftward motion &emsp;&emsp;&emsp;&emsp; (iv) Rightward motion
<img src="https://github.com/user-attachments/assets/424e7642-6287-4aa8-9a41-1e6ad27efb7e" width="200" height="200"/>
<img src="https://github.com/user-attachments/assets/5f020d86-c73c-4ddc-a0dd-8eceffe74d2b" width="200" height="200"/>
<img src="https://github.com/user-attachments/assets/437b1498-d7fa-4db3-be33-aff256ef7fd0" width="200" height="200"/>
<img src="https://github.com/user-attachments/assets/a5861f71-0ac9-4479-8c05-ab8ac25f3630" width="200" height="200"/>

#### (v) Anti-Clockwise Rotation &emsp;&emsp; (vi) Clockwise Rotation
<img src="https://hackmd.io/_uploads/HJr1hVH_C.gif" width="200" height="200"/> <img src="https://hackmd.io/_uploads/Bkhg2VB_A.gif" width="200" height="200"/>

#### (vii) Forward-right motion &emsp;&emsp; (viii) Backward-right motion &emsp;&emsp; (ix) Forward-left motion &emsp;&emsp; (x) Backward-left motion
<img src="https://hackmd.io/_uploads/rJsTrrHOA.gif =200x" width="200" height="200"/> <img src="https://hackmd.io/_uploads/HyqAHHBdC.gif" width="200" height="200"/>
<img src="https://hackmd.io/_uploads/HyEJISHdA.gif" width="200" height="200"/> <img src="https://hackmd.io/_uploads/SkKJUHHuR.gif" width="200" height="200"/>

#### Motion of robot with wheel rotating direction
<img src="https://hackmd.io/_uploads/S1BWw0Bfyx.jpg" alt="Image" style="height: 500px; width: 500;">

### 4. Speed Control with PWM Pins
- **Purpose** : To control the speed of the motors driving the Mecanum wheels, we employed Pulse Width Modulation (PWM). This technique allows us to simulate varying levels of voltage by rapidly switching a digital signal between high (on) and low (off) states.
- **How PWM Pins Works** :
  -  The PWM signals are sent to the L298N motor driver through its enable pins (ENA and ENB). By adjusting these signals, we can finely control the speed of each motor based on user gestures.
- **Procedure** : 
  - Used the ESP32 PWM pins to generate Pulse Width Modulation (PWM) signals for the L298N motor driver.
  - Maped gesture intensities to speed levels (e.g., tilt angle to PWM duty cycle).
  - Ensure smooth acceleration and deceleration by fine-tuning the PWM signals.
- **Outcome** : Enables precise control of motor speeds, ensuring the robot moves efficiently and responsively.
- ### 5. Wireless Data Transfer using WiFi (ESP-NOW Protocol)

- **Procedure**:
  - Used ESP32 modules for both transmitting and receiving data.
  - Implemented the ESP-NOW protocol, which allows peer-to-peer communication with low latency.
  - Transmited gesture data from the glove as control commands (e.g., direction, speed).
![image =470x400](https://github.com/user-attachments/assets/5da94b60-d904-4400-b095-26e75b552fd4)

#### Circuit of Hand-Gesture IMU :
![Screenshot (131)](https://hackmd.io/_uploads/ryevkZnJke.png)

#### Circuit of Omni-Directional Robot:
![Screenshot (132)](https://hackmd.io/_uploads/B1jEy-n1Je.png)
## Final Outcome

### 1] Forward Motion and Gesture
| <img src="https://hackmd.io/_uploads/r1yJAIdM1l.gif" alt="Image 1" style="height: 300px; width: 350px;"> | <img src="https://github.com/user-attachments/assets/f961f6bf-daa5-4100-b5eb-6523452671d6" style="height: 300px; width: 300px;"> |
| --- | --- |

### 2] Backward Motion and Gesture 
| <img src="https://hackmd.io/_uploads/rJgJG9uM1g.gif" alt="Image 1" style="height: 300px; width: 350px;"> | <img src="https://hackmd.io/_uploads/rymMTcPG1l.gif" alt="Image 2" style="height: 300px; width: 300px;"> |
| --- | --- |

### 3] Leftward motion and Gesture
| <img src="https://github.com/user-attachments/assets/dc1f4fce-7738-4a88-85a0-ec11f394c3f2" alt="Image 1" style="height: 300px; width: 350px;"> | <img src="https://github.com/user-attachments/assets/9730379a-bff3-45e2-8800-1b07a8ac1b5f" alt="Image 2" style="height: 300px; width: 300px;"> |
| --- | --- |

### 4] Rightward motion and Gesture
| <img src="https://github.com/user-attachments/assets/4b077c4d-f382-4126-a5b3-8cc5574c0782" alt="Image 1" style="height: 300px; width: 350px; transform: rotate(90deg);"> | <img src="https://hackmd.io/_uploads/ryT8bhvMyx.gif" alt="Image 2" style="height: 300px; width: 300px;"> |
| --- | --- |

### 5] Backward-Right motion and Gesture
| <img src="https://hackmd.io/_uploads/ByvxLhvzye.gif" alt="Image 1" style="height: 300px; width: 350px;"> | <img src="https://hackmd.io/_uploads/rk94Snvz1g.gif" alt="Image 2" style="height: 300px; width: 300px;"> |
| --- | --- |

### 6] Forward-Right motion and Gesture
| <img src="https://hackmd.io/_uploads/HyyDs8_Gyl.gif" alt="Image 1" style="height: 300px; width: 350px;"> | <img src="https://hackmd.io/_uploads/rJCcSTPz1g.gif" alt="Image 2" style="height: 300px; width: 300px;"> |
| --- | --- |

### 7] Backward-Left motion and Gesture
| <img src="https://hackmd.io/_uploads/B1ZfN8df1x.gif" alt="Image 1" style="height: 300px; width: 350px;"> | <img src="https://hackmd.io/_uploads/BkZywTvzJe.gif" alt="Image 2" style="height: 300px; width: 300px;"> |
| --- | --- |

### 8] Forward-Left motion and Gesture
| <img src="https://hackmd.io/_uploads/HJX4YU_Gkg.gif" alt="Image 1" style="height: 300px; width: 350px;"> | <img src="https://hackmd.io/_uploads/r1DS8I_zJx.gif" alt="Image 2" style="height: 300px; width: 300px;"> |
| --- | --- |

### 9] Clockwise motion and Gesture
| <img src="https://hackmd.io/_uploads/SymdFS_M1x.gif" alt="Image 1" style="height: 300px; width: 350px;"> | <img src="https://hackmd.io/_uploads/rJFf_B_fJx.gif" alt="Image 2" style="height: 300px; width: 300px;"> |
| --- | --- |

### 10] Anti-Clockwise motion and Gesture
| <img src="https://hackmd.io/_uploads/HJeEAHOf1x.gif" alt="Image 1" style="height: 300px; width: 350px;"> | <img src="https://hackmd.io/_uploads/Sy-RhBdfkg.gif" alt="Image 2" style="height: 300px; width: 300px;"> |
| --- | --- |

Link to extensive breakdown of project: https://hackmd.io/@r-D0tnDWS9yZpWf3XtGu4g/S1byF9wJJx
