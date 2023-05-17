import rclpy
from rclpy.node import Node
import sys
import time
import os

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Transform
from geometry_msgs.msg import Quaternion
from ackermann_msgs.msg import AckermannDriveStamped
from tf2_ros import TransformBroadcaster

from ament_index_python.packages import get_package_share_directory

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
        self.ref_traj = np.load(os.path.join(get_package_share_directory('f1tenth_gym_ros'),
                                            'resource',
                                            'ref_traj.npy'))
        self.ref_traj # prevent unused variable warning
        
        # create a timer to publish the control input every 20ms
        self.get_logger().info("Creating Timer")
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.timer # prevent unused variable warning
        
        self.pose = np.zeros(3)
        #####------------ADDAD BY HAREL---------------#####
        self.abs_cross_track_accumulated_error = 0
        self.abs_along_track_accumulated_error = 0
        self.diff_cross_error = 0
        self.diff_along_error = 0
        self.speed = 0
        self.delta = 0
        #####------------ADDAD BY HAREL--------END----#####
        self.current_cross_track_error = 0
        self.current_along_track_error = 0
        self.cross_track_accumulated_error = 0
        self.along_track_accumulated_error = 0
        self.waypoint_index = 0
        self.u0 = np.array([0, 1])
        
        self.moved = False
    
    def get_ref_pos(self):
        # get the next waypoint in the reference trajectory based on the current time
        waypoint = self.ref_traj[self.waypoint_index % len(self.ref_traj)]
        self.waypoint_index += 1
        return waypoint 

    def log_accumulated_error(self):
        ref_pos = self.get_ref_pos()
        
        cross_track_error = 0
        along_track_error = 0
        # compute the cross track and along track error depending on the current pose and the next waypoint
        #### YOUR CODE HERE ####

        ref_next_pose =  self.ref_traj[self.waypoint_index % len(self.ref_traj)]
        
        ref_theta = np.arctan2(ref_next_pose[1]-ref_pos[1],ref_next_pose[0]-ref_pos[0])

        cross_track_error = -np.sin(ref_theta) * (self.pose[0] - ref_pos[0]) + np.cos(ref_theta) *  (self.pose[1] - ref_pos[1])
        along_track_error = np.cos(ref_theta) * (self.pose[0] - ref_pos[0]) + np.sin(ref_theta) *  (self.pose[1] - ref_pos[1])
        
        
        #### END OF YOUR CODE ####
        self.diff_cross_error = cross_track_error - self.current_cross_track_error 
        self.diff_along_error = along_track_error - self.current_along_track_error
        self.current_cross_track_error = cross_track_error
        self.current_along_track_error = along_track_error
        
        # log the accumulated error to screen and internally to be printed at the end of the run
        #####------------CHANGED BY HAREL---------------#####
        self.get_logger().info("Cross Track Error: " + str(cross_track_error))
        self.get_logger().info("Along Track Error: " + str(along_track_error))
        self.cross_track_accumulated_error += cross_track_error
        self.along_track_accumulated_error += along_track_error
        self.abs_cross_track_accumulated_error += abs(cross_track_error)
        self.abs_along_track_accumulated_error += abs(along_track_error)
        self.get_logger().info("Cross Track Accumulated Error: " + str(self.abs_cross_track_accumulated_error))
        self.get_logger().info("Along Track Accumulated Error: " + str(self.abs_along_track_accumulated_error))
        #####------------CHANGED BY HAREL--------END----#####

    def odom_callback(self, msg):
        # get the current pose
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, yaw = euler.quat2euler([q.w, q.x, q.y, q.z])
        
        if not self.moved and (x < -1 and y > 3):
            self.moved = True
        elif self.moved and x > 0:
            raise EndLap
            
        
        self.pose = np.array([x, y, yaw])
        
    def timer_callback(self):
        self.log_accumulated_error()
        
        # compute the control input
        if self.controller == "pid_unicycle":
            u = self.pid_unicycle_control(self.pose)
        elif self.controller == "pid":
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
        #### YOUR CODE HERE ####
        """
        The akerman model from class is  :   [   x_dot   ]       [  speed*cos(theta)          ]
                                             [   y_dot   ]   =   [  speed*sin(theta)          ]    
                                             [ theta_dot ]       [ (speed/d)*tan(lambda)      ]

        to develop a PID we will write the model error   :

        we will seperate the error to 3:
            cross track error - calculated in previous section
            along track error - calculated in previous section
            orientation error -will shown below
            speed = K(cross track error)
            delta = K(orientation error, along track error)
            orientation_error = theta_error = theta_pos - theta_ref
            orientation_error_dot = theta_error_dot = theta_pos_dot = (speed/d)*tan(lambda)

        """

        # Length of car d taken from efo_racecar.xacro
        d = 0.3302 
        # step 1: calculate the Theta reference and Theta error
        
        
        ref_traj_current_step = self.ref_traj[self.waypoint_index % len(self.ref_traj)] #Get r_ref (the waypoint moved forward already for next step)
        try:
            ref_traj_next_step = self.ref_traj[(self.waypoint_index+2) % len(self.ref_traj)] #Get r_(ref+1) (the waypoint moved forward already for next step)
        except: ref_traj_next_step = ref_traj_current_step
        
        r_dot_ref = (ref_traj_next_step - ref_traj_current_step)  # r_ref_dot = r_(ref+2) - ref
        x_dot = r_dot_ref[0] # r_dot is a vector of x,y
        y_dot = r_dot_ref[1]
        pose2_trimmed = pose[2]
        theta_ref = np.arctan2(y_dot,x_dot)
        if pose[2] < 0:
            pose2_trimmed = 2*np.pi - np.abs(pose[2]) 
        if pose[2] == 0 and theta_ref ==np.pi:
            pose2_trimmed = np.pi

        theta_err = pose2_trimmed - theta_ref
        theta_err= np.arctan2(np.sin(theta_err), np.cos(theta_err))
        #Applying model difference:
        theta_error_dot = self.speed/ d * np.tan(self.delta) 
        

        Kp = 0.5
        Ki = 0.2#1
        Kd = 0.0#1
 
        # Kp2 = 0.8
        # Ki2 = 0.3
        # Kd2 = 0.5
        # # WORKS:
        Kp2 = 0.8
        Ki2 = 0.5
        Kd2 = 0.3

        #THE CONTROLLER:
        self.speed = -(Kp*self.current_along_track_error+Ki*self.along_track_accumulated_error+ Kd*self.diff_along_error) #PI
        self.delta = -(Kp2*(theta_err+self.current_cross_track_error)+Ki2*self.cross_track_accumulated_error + Kd2*(theta_error_dot))
        self.delta -= pose[2]
        steering_angle = self.delta
        steering_angle= np.arctan2(np.sin(steering_angle), np.cos(steering_angle))
  
        speed = self.speed
        return np.array([steering_angle, speed])
        
        #### END OF YOUR CODE ####
        raise NotImplementedError
    
    def pid_unicycle_control(self, pose):
        #### YOUR CODE HERE ####
        """
        
        The unicycle model from class is  : [   x_dot   ]       [speed*cos(theta) ]
                                            [   y_dot   ]   =   [speed*sin(theta) ]    
                                            [ theta_dot ]       [      delta      ]

        to develop a PID we will write the model error   :

        we will seperate the error to 3:
            cross track error - calculated in previous section
            along track error - calculated in previous section
            orientation error -will shown below
            speed = K(cross track error)
            delta = K(orientation error, along track error)
            orientation_error = theta_error = theta_pos - theta_ref
            orientation_error_dot = theta_error_dot = theta_pos_dot = delta
            
        """
        
        # step 1: calculate the Theta reference and Theta error

        ref_traj_current_step = self.ref_traj[self.waypoint_index % len(self.ref_traj)] #Get r_ref
        try:
            ref_traj_next_step = self.ref_traj[(self.waypoint_index+2) % len(self.ref_traj)] #Get r_(ref+2) (the waypoint moved forward already for next step)
        except: ref_traj_next_step = ref_traj_current_step
        
        r_dot_ref = (ref_traj_next_step - ref_traj_current_step)  # r_ref_dot = r_(ref+1) - ref
        x_dot = r_dot_ref[0] # r_dot is a vector of x,y
        y_dot = r_dot_ref[1]
        pose2_trimmed = pose[2]
        theta_ref = np.arctan2(y_dot,x_dot) #theta_ref = function of the y_dot,x_dot of the r_ref
        if pose[2] < 0:
            pose2_trimmed = 2*np.pi - np.abs(pose[2]) 
        if pose[2] == 0 and theta_ref ==np.pi:
            pose2_trimmed = np.pi
        theta_err = pose2_trimmed - theta_ref #calculate the error of the theta
        theta_err= np.arctan2(np.sin(theta_err), np.cos(theta_err))
        
        #Applying model difference:
        theta_err_dot = self.delta 

        Kp = 0.5
        Ki = 0.2#1

        Kp2 = 0.7
        Ki2 = 0.5#1
        Kd2 = 0.4#1
        speed = -(Kp*self.current_along_track_error+Ki*self.along_track_accumulated_error) #PI
        #THE CONTROLLER: 
        self.delta = -(Kp2*(self.current_cross_track_error+theta_err) +Ki2*self.cross_track_accumulated_error+ Kd2*theta_err_dot) #PID
        self.delta -= pose[2] # limit the angle of turn
        self.delta = np.arctan2(np.sin(self.delta), np.cos(self.delta))
        steering_angle = self.delta

        return np.array([steering_angle, speed])
        
        # return np.array([steering_angle, speed])
        #### END OF YOUR CODE ####
        raise NotImplementedError
    
    def pure_pursuit_control(self, pose):
        #### YOUR CODE HERE ####
        
        """
        Pure pursuit:
            steering_angle  = delta
            speed = ni = x_dot/cos(theta) = y_dot/sin(theta)
            delta = arctan2(2d*sin(alpha)/L)
            L = distance to target = sqrt(||pose -ref||**2)
            alpha = angle between the car's heading direction
                    and the line from the current pose to target
                  = arctan2(yt - y, xt - x) - theta
        """
        # calculate car speed
      
        Kp = 1.3
        Ki = 0.02
        ni = -(Kp*self.current_along_track_error+Ki*self.along_track_accumulated_error) #PI
        # Assume length of car is d = 1
        d = 0.3032

        # we will look four steps ahead
        try:
            ref_traj_next_4th_step = self.ref_traj[(self.waypoint_index+3) % len(self.ref_traj)] 
        except: ref_traj_next_4th_step = self.ref_traj[self.waypoint_index % len(self.ref_traj)]
        
        L = np.sqrt((pose[0]-ref_traj_next_4th_step[0])**2 +(pose[1]-ref_traj_next_4th_step[1])**2 )
        # self.get_logger().info("L is " + str(L) )
        alpha = np.arctan2((ref_traj_next_4th_step[1]-pose[1]),(ref_traj_next_4th_step[0]-pose[0]))
        #limit alpha to range:
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha)) - pose[2]
        #calculate the steering wheel:
        delta = np.arctan2(2*d*np.sin(alpha),L) 
        steering_angle = delta 
        speed = ni
        
        return np.array([steering_angle, speed])
        
        #### END OF YOUR CODE ####
        raise NotImplementedError

 # Define the cost function
    
    def cost(self,x, u, x_ref, y_ref, theta_ref, error_x, error_y, error_theta, k1, k2, k3, k4, k5):
        cost = (x[0] - x_ref)**2 + (x[1] - y_ref)**2 + (x[2] - theta_ref)**2 + k1*error_x**2 + k2*error_y**2 + k3*error_theta**2 + k4*u[0]**2 + k5*u[1]**2
        return cost

# Define the dynamics model

    def dynamics(self,x, u, L):
        x_dot = u[1]*np.cos(x[2])
        y_dot = u[1]*np.sin(x[2])
        theta_dot = (u[1]/L)*np.tan(u[0])
        x_next = x + np.array([x_dot, y_dot, theta_dot])
        return x_next

    def ilqr(self,x0, u0, x_ref, y_ref, theta_ref, error_x, error_y, error_theta, k1, k2, k3, k4, k5, L, max_iter=100, eps=1e-6):
        # Initialize the state and control inputs
        x = x0
        u = u0
        alpha = 1.0

        # Iterate until convergence
        for i in range(max_iter):
            # Linearize the dynamics model around the current state and control inputs
            A = np.zeros((3,3))
            B = np.zeros((3,2))
            C = np.zeros((3,1))
            for j in range(10):
                x_next = self.dynamics(x, u, L)
                A = np.eye(3) + A*(x_next - x)/10
                self.get_logger().warning("A in loop is "+ str(A))
                B = B + (self.dynamics(x, u + np.array([0, alpha/10]), L).reshape(3,1) - self.dynamics(x, u - np.array([0, alpha/10]), L).reshape(3,1))/alpha
                C = C + (self.cost(x_next, u + np.array([0, alpha/10]), x_ref, y_ref, theta_ref, error_x, error_y, error_theta, k1, k2, k3, k4, k5) - self.cost(x_next, u - np.array([0, alpha/10]), x_ref, y_ref, theta_ref, error_x, error_y, error_theta, k1, k2, k3, k4, k5))/alpha
            A = A/10
            B = B/10
            C = C/10

            # Use dynamic programming to calculate the optimal control inputs
            Q = np.zeros((3,3))
            R = np.zeros((2,2))
            q = np.zeros((3,1))
            r = np.zeros((2,1))
            K = np.zeros((2,3))
            k = np.zeros((2,1))
            V = np.zeros((3,1))
            v = np.zeros((1,1))
            for j in range(10):
                # self.get_logger().warning("Q is "+ str(Q))
                # self.get_logger().warning("A is "+ str(A))
                Q = A.T.dot(Q).dot(A) + np.eye(3)
                R = B.T.dot(Q).dot(B) + np.eye(2)*1e-6
                q = A.T.dot(q) + C
                r = B.T.dot(q)
                # self.get_logger().warning("R is "+ str(R))
                K = np.linalg.inv(R).dot(B.T).dot(Q)
                k = np.linalg.inv(R).dot(r)
                V = Q.dot(K.T)
                v = q.T.dot(K.T)
            K = K/10
            k = k/10
            V = V/10
            v = v/10

            # Update the state and control inputs using the optimal control inputs
            print("KKKKKK",k[0])
            u_next = u + K.dot((x - x0)) + k[0]
            x_next = self.dynamics(x, u_next, L)
            if np.linalg.norm(x_next - x) < eps:
                break
            x = x_next
            u = u_next

        # Return the optimal control inputs
        return u  

    def ilqr_control(self, pose):
        #### YOUR CODE HERE ####
        d = 0.3302 
        L = 0.2032
        if pose[0] == 0 and pose[1] ==0:
            self.get_logger().warning("~~~~~~~~~BEGIN~~~~~~~~")
            self.last_u  =self.u0
        ref_traj = self.ref_traj[self.waypoint_index+3 % len(self.ref_traj)] 
        error_x = pose[0] -ref_traj[0]
        error_y = pose[1] -ref_traj[1]
        theta_ref = np.arctan2(ref_traj[1],ref_traj[0])
        error_theta = pose[2] -theta_ref
        k1 = 1.0
        k2 = 1.0
        k3 = 1.0
        k4 = 0.1
        k5 = 0.1
        u = self.ilqr(pose, self.last_u, ref_traj[0], ref_traj[1] ,theta_ref, error_x, error_y, error_theta, k1, k2, k3, k4, k5, L, max_iter=20, eps=1e-6)
        steering_angle = float(u[0])
        self.get_logger().warning("the steering angle is "+ str(steering_angle))
        speed = float(u[1])
        self.last_u = u
        return np.array([steering_angle, speed])
        #### END OF YOUR CODE ####
        raise NotImplementedError
        
    def optimal_control(self, pose):
        #### YOUR CODE HERE ####
        
        
        # return np.array([steering_angle, speed])
        #### END OF YOUR CODE ####
        raise NotImplementedError



# Implement the iLQR algorithm
    

class EndLap(Exception):
    # this exception is raised when the car crosses the finish line
    pass


def main(args=None):
    rclpy.init()

    lab1 = Lab1(controller_type=sys.argv[1])

    tick = time.time()
    try:
        rclpy.spin(lab1)
    except NotImplementedError:
        rclpy.logging.get_logger('lab1').info("You havn't implemented this controller yet!")
    except EndLap:
        tock = time.time()
        rclpy.logging.get_logger('lab1').info("Finished lap")
        rclpy.logging.get_logger('lab1').info("Cross Track Error: " + str(lab1.cross_track_accumulated_error))
        rclpy.logging.get_logger('lab1').info("Along Track Error: " + str(lab1.along_track_accumulated_error))
        rclpy.logging.get_logger('lab1').info("Cross Track Accumulated Error: " + str(lab1.abs_cross_track_accumulated_error))
        rclpy.logging.get_logger('lab1').info("Along Track Accumulated Error: " + str(lab1.abs_along_track_accumulated_error))
        rclpy.logging.get_logger('lab1').info("Lap Time: " + str(tock - tick))
        print("Cross Track Error: " + str(lab1.cross_track_accumulated_error))
        print("Along Track Error: " + str(lab1.along_track_accumulated_error))
        print("Lap Time: " + str(tock - tick))

    lab1.destroy_node()
    rclpy.shutdown()
    
    
if __name__ == '__main__':
    controller_type = sys.argv[1]
    main(controller_type)
    
    
        