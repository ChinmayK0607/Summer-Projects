#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist

def draw_square():
    rospy.init_node('draw_square', anonymous=True)
    velocity_publisher = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)
    rate = rospy.Rate(1)  # 1 Hz

    vel_msg = Twist()

    for _ in range(5):
        vel_msg.linear.x = 2.0 
        vel_msg.angular.z = 0
        velocity_publisher.publish(vel_msg)
        rospy.sleep(2)  # move for 2 seconds

        vel_msg.linear.x = 0
        vel_msg.angular.z = 1.57 
        velocity_publisher.publish(vel_msg)
        rospy.sleep(1)  # turn for 1 second

    vel_msg.linear.x = 0
    vel_msg.angular.z = 0
    velocity_publisher.publish(vel_msg)

if __name__ == '__main__':
    try:
        draw_square()
    except rospy.ROSInterruptException:
        pass
