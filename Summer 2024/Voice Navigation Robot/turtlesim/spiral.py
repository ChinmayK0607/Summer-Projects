#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist

def spiral_pattern():
    rospy.init_node('turtlesim_spiral', anonymous=True)
    pub = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)
    rate = rospy.Rate(10)  # 10hz
    vel_msg = Twist()

    linear_velocity = 0.5
    angular_velocity = 0.5

    while not rospy.is_shutdown():
        vel_msg.linear.x = linear_velocity
        vel_msg.angular.z = angular_velocity

        pub.publish(vel_msg)

        linear_velocity += 0.15
        angular_velocity += 0.07

        rate.sleep()
	
if __name__ == '__main__':
	try:
		spiral_pattern()
	except rospy.ROSInterruptException:
        	pass
	
	
