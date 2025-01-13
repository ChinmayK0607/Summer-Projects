#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist

move_cmd = Twist()

def callback(data):
    global move_cmd
    try:
        if data.data == 'FORWARD':
            move_cmd.linear.x = 0.2 
            move_cmd.angular.z = 0.0
        elif data.data == 'BACKWARD':
            move_cmd.linear.x = -0.2 
            move_cmd.angular.z = 0.0
        elif data.data == 'LEFT':
            move_cmd.linear.x = 0.0
            move_cmd.angular.z = 0.4  
        elif data.data == 'RIGHT':
            move_cmd.linear.x = 0.0
            move_cmd.angular.z = -0.4 
        elif data.data == 'STOP':
            move_cmd.linear.x = 0.0
            move_cmd.angular.z = 0.0
    except AttributeError:
        pass

def voice_cmd():
    rospy.init_node('voice_cmd', anonymous=True)
    pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
    rate = rospy.Rate(1) 

    while not rospy.is_shutdown():
        sub = rospy.Subscriber('speech', String, callback)
        pub.publish(move_cmd)
        rospy.loginfo(f"Received command : \n {move_cmd} \n")
        rate.sleep()

if __name__ == '__main__':
    try:
        voice_cmd()
    except rospy.ROSInterruptException:
        pass
