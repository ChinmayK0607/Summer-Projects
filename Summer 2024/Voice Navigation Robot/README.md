# Voice Navigation Robot

## Objective

To create a voice-controlled robot using ROS that can be controlled using voice commands to navigate and perform tasks.

## Requirements

* Ubuntu 20.04
* ROS Noetic
* Pocketsphinx
* Gstreamer

## How It Works
1. **Pipeline :** A Gstreamer pipeline is established to integrate the voice model and establish a modular communication between components.
2. **Recognition :** A ROS publisher node with the voice models listens to microphone input for voice input.
3. **Processing :** Upon detecting a command, the voice model converts the voice input into string data.
4. **Publishing :** The string data is published over a speech topic.
5. **Execution :** A node with a subscriber and a publisher, listens for the incoming data over the topic and publishes relevant commands to the robot.


## Turtlesim
Turtlesim is a basic, beginner-friendly simulator package in ROS that is designed for learning the basic concepts of ROS. It demonstrates the basic principles of ROS and gives you an overview of how ROS communications and systems are designed.

> 1.Square Spiral
### ![ezgif-2-9f758776e7](https://github.com/sangwan7gaurav/Voice_Navigation/assets/138971930/e48e4f3d-5b82-48fc-820c-ffeda10232f2) 
> 2.Circle
### ![2](https://github.com/sangwan7gaurav/Voice_Navigation/assets/138971930/afdc9c2d-79d7-4569-9bac-3cb6f46373ac)
> 3.Square
### ![1](https://github.com/sangwan7gaurav/Voice_Navigation/assets/138971930/4d06d620-a81b-4201-8106-b02912b38383)
> 4.Go to Goal
### ![ezgif-2-e18b684b14](https://github.com/sangwan7gaurav/Voice_Navigation/assets/138971930/97559c15-b698-4bae-aaad-434488b60987)



## Gazebo

Gazebo is a robotics simulation environment included within ROS. It provides a 3D physics-based simulation of robots, environments, and sensors, allowing users to simulate and test robotic systems in a virtual environment before deploying them in the real world.


![Screenshot_from_2024-07-08_19-38-48_optimized](https://github.com/user-attachments/assets/ebf896a3-99d9-4b0d-b8c7-bb94ffbdb0f3)


## Turtlebot 3

TurtleBot 3 is a versatile open source robot platform designed for education, research, and prototyping applications in robotics. It is fully compatible with ROS, making it easy to integrate with ROS-based software libraries, tools, and simulations allowing users to leverage the extensive ROS ecosystem for developing and testing robotic applications.


INSTALLATION:
1) Run the following commands in your terminal. Make sure to clone the repositories into your workspace.
   ```
   git clone https://github.com/ROBOTIS-GIT/turtlebot3.git
   git clone https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git
   git clone https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git
   ```
2) Build your packages.
   ```
   catkin_make
   ```

BASIC SIMULATION:
1) Export your turtlebot model. We will be using burger.
   ```
   export TURTLEBOT3_MODEL=burger
   ```
2) To launch and empty world use the following command.
   ```
   roslaunch turtlebot3_gazebo turtlebot3_empty_world.launch
   ```
3) To launch another world with objects use the following command.
   ```
   roslaunch turtlebot3_gazebo turtlebot3_world.launch
   ```
4) To perform teleoperations open another terminal and run the following.
   ```
   export TURTLEBOT3_MODEL=burger
   roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
   ```
![ezgif-1-2f630658bd](https://github.com/user-attachments/assets/0e1b9ee7-16bb-45a3-9458-2d36231953f0)


RESOURCES:

[Turtlebot 3 Manual](https://emanual.robotis.com/docs/en/platform/turtlebot3/overview/)<br>

## SLAM

![WhatsApp GIF 2024-12-14 at 19 39 52](https://github.com/user-attachments/assets/4aba1b13-c68b-4894-9edd-8c033a2fdcfd)

Integration of Simultaneous Localization and Mapping (SLAM) can enable seamless navigation in dynamic environments and plan paths in real-time. This synergy allows the robot to autonomously execute commands like "Go to the kitchen" or "Explore the room," with accurate obstacle avoidance and adaptive pathfinding. By leveraging SLAM, the robot ensures precise navigation even in unknown or changing environments, enhancing user interaction and overall functionality.

We explored two SLAM algorithms in our simulations, namely GMapping and HectorSLAM. GMapping and Hector SLAM serve different use cases in robotics based on their strengths. GMapping relies on particle filters and requires odometry data, making it well-suited for low-speed robots operating in static or semi-static environments, such as offices or warehouses. In contrast, Hector SLAM does not depend on odometry and uses scan matching for mapping, making it ideal for high-speed or dynamic applications, such as drones or robots without reliable wheel encoders. While GMapping excels in precise localization for slower robots, Hector SLAM is optimized for real-time mapping in high-motion scenarios.


## Gstreamer

GStreamer is a multimedia framework that provides a pipeline-based architecture for constructing multimedia applications. It is open source and widely used in various applications and platforms to handle multimedia processing, streaming, and playback. We will be using gstreamer to integrate pocketsphinx into our ROS project. Its modular nature will help us examine indivudual elements of the pipeline.

INSTALLATION:  

```
apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio
```

## Voice Models

### Pocketsphinx

PocketSphinx is a lightweight speech recognition engine developed by Carnegie Mellon University (CMU).

INSTALLATION:
1) Clone the [pocketsphinx](https://github.com/cmusphinx/pocketsphinx) and [sphinxbase](https://github.com/cmusphinx/sphinxbase) repostories into your workspaces.
2) Go into the sphinxbase folder and build it.

   ```
   ./autogen.sh
   ./configure
   make
   make install
   ```


### VOSK 

The VOSK voice model is another lightweight, open-source speech recognition toolkit that allows for offline voice command processing.

INSTALLATION:  

1. Run the following in a terminal.
```
pip install vosk
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
```

2. Load the model into your Vosk speech recognition code by specifying the path to the unzipped model directory.

RESOURCES:

[VOSK GitHub Repository](https://github.com/alphacep/vosk-api)<br>


### Why Two Models?

Both of these models provide vastly different benifits. Pocketsphinx is a lightweight library while VOSK is resource intensive. However setting up dictionaries and custom LM files for pocketsphinx is not as easy. Pocketsphinx is ideal for small vocabularies but VOSK has better accuracy in noisy environments. The ideal library is dependent on the task to be performed.
