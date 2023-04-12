import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Transform
from geometry_msgs.msg import Quaternion
from ackermann_msgs.msg import AckermannDriveStamped
from tf2_ros import TransformBroadcaster

import gym
import numpy as np
from transforms3d import euler

# this node should get the current pose from odom and get a reference trajectory from a yaml file
# and publish ackermann drive commands to the car based on one of 4 controllers selected with a parameter
# the controllers are PID, Pure Pursuit, iLQR, and an optimal controller 
class Lab1(Node):
    def __init__(self, controller_type: str = 'pid'):
        super().__init__('lab1')
        self.get_logger().info("Lab 1 Node has been started")

        # get parameters
        self.controller = self.declare_parameter('controller', controller_type).value
        self.get_logger().info("Controller: " + self.controller)
        # to set the parameter, run the following command in a terminal when running a different controller
        # ros2 run f1tenth_gym_ros lab1.py --ros-args -p controller:=<controller_type>

        # get the current pose
        self.get_logger().info("Subscribing to Odometry")
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.odom_sub # prevent unused variable warning
        
        self.get_logger().info("Publishing to Ackermann Drive")
        self.cmd_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.cmd_pub # prevent unused variable warning
        
        # get the reference trajectory
        self.get_logger().info("Loading Reference Trajectory")
        self.ref_traj = np.load('scripts/ref_traj.npy')
        self.ref_traj # prevent unused variable warning
        
        # create a timer to publish the control input every 20ms
        self.get_logger().info("Creating Timer")
        self.timer = self.create_timer(0.005, self.timer_callback)
        self.timer # prevent unused variable warning
        
        self.cross_track_accumulated_error = 0
        self.along_track_accumulated_error = 0
        self.waypoint_index = 0
    
    def get_timed_waypoint(self):
        # get the next waypoint in the reference trajectory based on the current time
        waypoint = self.ref_traj[self.waypoint_index % len(self.ref_traj)]
        self.waypoint_index += 1
        return waypoint

    def log_accumulated_error(self):
        timed_waypoint = self.get_timed_waypoint()
        
        # compute the cross track and along track error depending on the current pose and the next waypoint
        # if the ref point is at the top or bottom of the track, the cross track error is the y distance
        if timed_waypoint[1] == 9.5 or timed_waypoint[1] == -13.5:
            cross_track_error = np.abs(timed_waypoint[1] - self.pose[1])
            along_track_error = np.abs(timed_waypoint[0] - self.pose[0])
        else:
            cross_track_error = np.abs(timed_waypoint[0] - self.pose[0])
            along_track_error = np.abs(timed_waypoint[1] - self.pose[1])
        
        # log the accumulated error to screen and internally to be printed at the end of the run
        self.get_logger().info("Cross Track Error: " + str(cross_track_error))
        self.get_logger().info("Along Track Error: " + str(along_track_error))
        self.cross_track_accumulated_error += cross_track_error
        self.along_track_accumulated_error += along_track_error
        
    
    def odom_callback(self, msg):
        # get the current pose
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, yaw = euler.quat2euler([q.x, q.y, q.z, q.w])
        
        self.pose = np.array([x, y, yaw])
        
    def timer_callback(self):
        # compute the control input
        if self.controller == "pid":
            u = self.pid_control(self.pose)
        elif self.controller == "pure_pursuit":
            u = self.pure_pursuit_control(self.pose)
        elif self.controller == "ilqr":
            u = self.ilqr_control(self.pose)
        elif self.controller == "optimal":
            u = self.optimal_control(self.pose)
        else:
            self.get_logger().info("Unknown controller")
            return
        
        # publish the control input
        cmd = AckermannDriveStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "base_link"
        cmd.drive.steering_angle = u[0]
        cmd.drive.speed = u[1]
        self.cmd_pub.publish(cmd)

    def pid_control(self, pose):
        raise NotImplementedError
        #### YOUR CODE HERE ####
        
        
        # return np.array([steering_angle, speed])

        #### END OF YOUR CODE ####
    
    def pure_pursuit_control(self, pose):
        raise NotImplementedError
        #### YOUR CODE HERE ####
        
        
        # return np.array([steering_angle, speed])

        #### END OF YOUR CODE ####
        
    def ilqr_control(self, pose):
        raise NotImplementedError
        #### YOUR CODE HERE ####
        
        
        # return np.array([steering_angle, speed])

        #### END OF YOUR CODE ####
        
    def optimal_control(self, pose):
        raise NotImplementedError
        #### YOUR CODE HERE ####
        
        
        # return np.array([steering_angle, speed])

        #### END OF YOUR CODE ####
        
def main(args=None):
    rclpy.init(controller_type=args)

    lab1 = Lab1()

    rclpy.spin(lab1)

    lab1.destroy_node()
    rclpy.shutdown()
    
    
if __name__ == '__main__':
    main('pid')
    # main('pure_pursuit')
    # main('ilqr')
    # main('optimal')
    
    
        