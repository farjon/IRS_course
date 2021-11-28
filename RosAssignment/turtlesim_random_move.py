#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
import math
import time
import numpy as np

x=0
y=0
z=0
yaw=0

def poseCallback(pose_message):
    global x
    global y, z, yaw
    x= pose_message.x
    y= pose_message.y
    yaw = pose_message.theta

def move():
    velocity_message = Twist()
    x0=x
    y0=y
    velocity_message.linear.x = 1.0
    velocity_message.angular.z = np.random.rand() *3
    distance_moved = 0.0
    loop_rate = rospy.Rate(10) 
    cmd_vel_topic='/turtle1/cmd_vel'
    velocity_publisher = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)

    while True :
            rospy.loginfo("Turtlesim moves forwards")
            velocity_publisher.publish(velocity_message)

            loop_rate.sleep()
            distance_moved = distance_moved+abs(0.5 * math.sqrt(((x-x0) ** 2) + ((y-y0) ** 2)))
            if  not (distance_moved<5.0):
                rospy.loginfo("reached")
                break
    
    velocity_message.linear.x =0
    velocity_publisher.publish(velocity_message)

if __name__ == '__main__':
    rospy.init_node('turtlesim_random_move')
    position_topic = "/turtle1/pose"
    pose_subscriber = rospy.Subscriber(position_topic, Pose, poseCallback) 
    while True:
        time.sleep(3)
        print('move: ')
        move()

