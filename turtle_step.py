#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose

import numpy as np

class TurtleMover():
    def __init__(self):
        rospy.init_node('turtle_mover')
        self.pub = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=1)
        self.sub = rospy.Subscriber('/turtle1/pose', Pose, callback=self.pose_callback)
        self.rate = rospy.Rate(50)
        self.time_limit = 0.2
        self.current_pose = None
        self.action_to_command = {
            0: [0.0, 0.0],
            1: [2.0, 0.0],
            2: [0.0, 2.0],
            3: [0.0, -2.0]
        }
        while self.current_pose is None:
            rospy.sleep(0.05)

    def pose_callback(self, pose):
        self.current_pose = pose

    def move(self, action):
        msg = Twist()
        t0 = rospy.Time.now().to_sec()
        stop = False
        while not stop:
            t = rospy.Time.now().to_sec()
            if (t - t0) < self.time_limit:
                msg.linear.x, msg.angular.z = self.action_to_command[action]
            else:
                msg.linear.x, msg.angular.z = self.action_to_command[0]
                stop = True
            self.pub.publish(msg)
            self.rate.sleep()
        return [self.current_pose.x, self.current_pose.y, self.current_pose.theta]


