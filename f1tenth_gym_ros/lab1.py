import rclpy
from rclpy.node import Node
import sys
import time
import os
import matplotlib.pyplot as plt

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
        
        self.current_cross_track_error = 0
        self.current_along_track_error = 0
        self.cross_track_accumulated_error = 0
        self.along_track_accumulated_error = 0
        self.waypoint_index = 0

        # Historians for PID-controllers
        self.previous_cross_track_error = 0
        self.previous_along_track_error= 0
        self.integral_along_track_error = 0
        self.integral_cross_track_error = 0

        self.integral_e_theta = 0
        self.previous_e_theta = 0
        self.integral = np.zeros((2,1,1))

        self.previous_pose = np.zeros((3,1))
        self.previous_error = np.zeros((3,1,1))


        self.x_plot = np.zeros((10 ,252, 3,1))

        self.previous_index = 0
    
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

        # Compute theta_ref
        wp_number = self.waypoint_index; 
        current_wp = self.ref_traj[wp_number % len(self.ref_traj)]
        next_wp = self.ref_traj[(wp_number+1) % len(self.ref_traj)]

        theta_ref = np.arctan2(next_wp[1]-current_wp[1], next_wp[0]-current_wp[0])

        # Compute errors acoording to lecture formula       
        #e_ct = -sin(theta_ref)* (x_ref) - x)* + cos(theta_ref)* (y-y_ref)
        cross_track_error = -np.sin(theta_ref) * (self.pose[0]-current_wp[0]) + np.cos(theta_ref)*((self.pose[1]-current_wp[1]))
        
        # e_at = -cos(theta_ref)*(x-x_ref) + sin(theta_ref)*(y-y_ref)
        along_track_error = np.cos(theta_ref)*(self.pose[0]-current_wp[0]) +  np.sin(theta_ref)*((self.pose[1]-current_wp[1]))

        #### END OF YOUR CODE ####
        self.current_cross_track_error = cross_track_error
        self.current_along_track_error = along_track_error
        
        # log the accumulated error to screen and internally to be printed at the end of the run
        # self.get_logger().info("Cross Track Error: " + str(cross_track_error))
        # self.get_logger().info("Along Track Error: " + str(along_track_error))
        self.cross_track_accumulated_error += abs(cross_track_error)
        self.along_track_accumulated_error += abs(along_track_error)
        
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

        # Preallocate variables
        u = np.zeros((2,1,1))
        e = np.zeros((3,1,1))
        e_dot = np.zeros((3,1,1))
        proportional = np.zeros((2,1,1))
        derivative = np.zeros((2,1,1))

        # Tuning
        Kp = np.array([[-1,0,5e-2],
                       [0,1,0]])
        Ki = np.array([[0,0,5e-3],
                       [0,1e-5,0]])
        Kd = np.array([[-10,0,1e-2],
                       [0,1,0]])

        # Parameters
        dt = 0.5 # Simulation timesteps
        d = 0.0381 # wheel-lenght of the car
        max_speed = 0.5 # Output limit according to ROS simulation (0.5 m/s)
        max_angle = 10*np.pi # Output limit according to a "normal car" -> steering angle >90deg is not feasible

        # Get reference trajectory
        current_wp = self.ref_traj[self.waypoint_index % len(self.ref_traj)]
        next_wp = self.ref_traj[(self.waypoint_index+1) % len(self.ref_traj)]
        # Compute reference theta
        theta_ref = np.arctan2(next_wp[1]-current_wp[1], next_wp[0]-current_wp[0])

        # Error term
        # Project the yaw to full angle range (-2pi, pi)
        yaw_projected = pose[2]
        if theta_ref >= 3.12 and theta_ref <= 3.16 and pose[2]<0:
            yaw_projected = 2*np.pi+pose[2]
        else:
            yaw_projected = pose[2]

        theta_e = np.clip(theta_ref - yaw_projected, -2*np.pi, 2*np.pi)

        # Compute error terms
        e[0,0,0] = theta_e
        e[1,0,0] = self.current_along_track_error
        e[2,0,0] = self.current_cross_track_error
        e_dot[:,:,0] = (self.previous_error[:,:,0] - e_dot[:,:,0] ) /dt
        
        proportional[:,:,0] = Kp @ e[:,:,0]

        # Compute integral and derivative terms
        self.integral[:,:,0] += Ki @ e[:,:,0] * dt
        self.integral[0,0,0] = np.clip(self.integral[0,0,0],-max_angle, max_angle,)  # Avoid integral windup
        self.integral[1,0,0] = np.clip(self.integral[1,0,0],-max_speed, max_speed,)  # Avoid integral windup

        if self.waypoint_index == 0: 
            # no previous errror
            derivative[:,:,0] = Kd @ e[:,:,0]
        else:
            derivative[:,:,0] = Kd @ e_dot[:,:,0] / dt

        # Compute final output
        u[:,:,:] = -(proportional + self.integral + derivative)
        u[0,0,0] = np.clip(u[0,0,0], -max_angle, max_angle) 
        u[1,0,0] = np.clip(u[1,0,0], -max_speed, max_speed) 

        # Keep track of state
        self.previous_pose = pose
        # Keep track of errors
        self.previous_error = e

        # Ackermann
        u[0,0,0] = np.arctan((u[0,0,0] * d)/u[1,0,0])        
        return np.array([u[0,0,0], u[1,0,0]])
        #### END OF YOUR CODE ####

        raise NotImplementedError
    
    def pid_unicycle_control(self, pose):
        #### YOUR CODE HERE ####
        
        # Preallocate variables
        u = np.zeros((2,1,1))
        e = np.zeros((3,1,1))
        e_dot = np.zeros((3,1,1))
        proportional = np.zeros((2,1,1))
        derivative = np.zeros((2,1,1))

        # Tuning
        Kp = np.array([[-1,0,5e-2],
                       [0,1,0]])
        Ki = np.array([[0,0,5e-3],
                       [0,1e-5,0]])
        Kd = np.array([[-1e-1,0,1e-2],
                       [0,1,0]])

        # Parameters
        dt = 0.5 # Simulation timesteps
        d = 0.018 # TODO TODO wheel-lenght of the car
        max_speed = 0.5 # Output limit according to ROS simulation (0.5 m/s)
        max_angle = 1.57 # Output limit according to a "normal car" -> steering angle >90deg is not feasible

        # Get reference trajectory
        current_wp = self.ref_traj[self.waypoint_index % len(self.ref_traj)]
        next_wp = self.ref_traj[(self.waypoint_index+1) % len(self.ref_traj)]
        # Compute reference theta
        theta_ref = np.arctan2(next_wp[1]-current_wp[1], next_wp[0]-current_wp[0])

        # Error term
        # Project the yaw to full angle range (-2pi, pi)
        yaw_projected = pose[2]
        if theta_ref >= 3.12 and theta_ref <= 3.16 and pose[2]<0:
            yaw_projected = 2*np.pi+pose[2]
        else:
            yaw_projected = pose[2]

        theta_e = np.clip(theta_ref - yaw_projected, -2*np.pi, 2*np.pi)

        # Compute error terms
        e[0,0,0] = theta_e
        e[1,0,0] = self.current_along_track_error
        e[2,0,0] = self.current_cross_track_error
        e_dot[:,:,0] = (self.previous_error[:,:,0] - e_dot[:,:,0] ) /dt
        
        proportional[:,:,0] = Kp @ e[:,:,0]

        # Compute integral and derivative terms
        self.integral[:,:,0] += Ki @ e[:,:,0] * dt
        self.integral[0,0,0] = np.clip(self.integral[0,0,0],-max_angle, max_angle,)  # Avoid integral windup
        self.integral[1,0,0] = np.clip(self.integral[1,0,0],-max_speed, max_speed,)  # Avoid integral windup

        if self.waypoint_index == 0: 
            # no previous errror
            derivative[:,:,0] = Kd @ e[:,:,0]
        else:
            derivative[:,:,0] = Kd @ e_dot[:,:,0] / dt

        # Compute final output
        u[:,:,:] = -(proportional + self.integral + derivative)
        u[0,0,0] = np.clip(u[0,0,0], -max_angle, max_angle) 
        u[1,0,0] = np.clip(u[1,0,0], -max_speed, max_speed) 

        # Keep track of state
        self.previous_pose = pose
        # Keep track of errors
        self.previous_error = e

        # Unicycle
        u[0,0,0] = u[0,0,0]
        
        return np.array([u[0,0,0], u[1,0,0]])
        #### END OF YOUR CODE ####
        raise NotImplementedError

    def pure_pursuit_control(self, pose):
        #### YOUR CODE HERE ####
        speed = 0.5 # set speed to a constant for a easy intro / debugging

        ## PURE PURSUIT CONTROLLER FOR steering angle 
        ## Paremeters
        d = 0.3302 # [m] axle distance
        L = 1 # [m] look ahead distance
        updated = False # sanity check

        # Reset if trajectory is nearly done -> so the controller can run until the end / multiple times
        if self.previous_index == 250:
            self.previous_index = 0

        # Find (x_r, y_r) which is the intersecection of a circle (L) and the reference trajectory
        for i in range(self.previous_index,len(self.ref_traj),1):
            candidate = self.ref_traj[i] 
            distance = np.linalg.norm(candidate-pose[0:2])
            if distance >= L: # we choose the first point along the trajectory that is more then L away. (to always comply with the min. look-ahead distance)
                x_r = candidate[0]
                y_r = candidate[1]
                self.previous_index = i
                updated = True
                break

        if not updated:
            print("No reference waypoint found!")

        # Calculate steering angle
        alfa =  np.arctan2(y_r - pose[1], x_r - pose[0]) -pose[2]
        steering_angle = np.arctan((2*d*np.sin(alfa))/L)
        
        return np.array([steering_angle, speed])
        #### END OF YOUR CODE ####
        raise NotImplementedError

    def ilqr_control(self, pose):
        #### YOUR CODE HERE ####

        def linearize_dynamics(v_i, delta_i, theta_i, x_it1, y_it1, theta_it1): 
            # Calculate the linearized state-space equation based on the current u and state

            # Linearization based on jacobian linearization
            A_t = np.array([[1, 0, -v_i *np.sin(theta_i)], [0, 1, v_i * np.cos(theta_i)], [0, 0, 1]])
            B_t = np.array([[np.cos(theta_i), 0],[np.sin(theta_i), 0 ], [np.tan(delta_i)/d, v_i/(d* np.square(np.cos(delta_i))) ]])
            f_x_xt1 = np.array([[v_i*np.cos(theta_i) - x_it1], [v_i*np.sin(theta_i) - y_it1], [v_i*np.tan(delta_i)/d - theta_it1]])

            # Represent in homogenous coordinate systems
            A_ht = np.concatenate((A_t,f_x_xt1), axis=1)
            A_ht = np.append(A_ht, [[0,0,0,1]], axis=0)
            B_ht = np.concatenate((B_t, np.array([[0,0,0]]).T), axis=1)
            B_ht = np.append(B_ht, [[0,0,1]], axis=0)

            return A_ht, B_ht     
        
        def add_theta(traj_x):
            # Calculate theta_ref based on the tangent between the current and the next waypoint
            for j in range(0,N-1,1):
                traj_x[0,j,2] =  np.arctan2(traj_x[0,j+1,1]-traj_x[0,j,1], traj_x[0,j+1,0]-traj_x[0,j,0])
            return traj_x       

        def get_Q_hom(traj_x_ref, Q, i, t):
            Q_hom_12 = Q @ np.array([traj_x[i,t,:,0]-traj_x_ref[i,t,:,0]]).T  
            # = Q(x_t^i - x_t^ref)
 
            Q_hom_21 = np.array([traj_x[i,t,:,0]-traj_x_ref[i,t,:,0]]) @ Q
            # = (x_t^i - x_t^ref)^T Q

            Q_hom_22 = (traj_x[i,t,:,0]-traj_x_ref[i,t,:,0]) @ np.array([traj_x[i,t,:,0]-traj_x_ref[i,t,:,0]]).T
            # = (x_t^i - x_t^ref)^T(x_t^i - x_t^ref)            

            return np.concatenate((np.vstack((Q,Q_hom_21)),np.vstack((Q_hom_12,Q_hom_22))), axis=1)
        
        ### MAIN ### 
        current_timestep = self.waypoint_index
        self.get_ref_pos()
        # Parameters
        s_dim = 3 # dimension of the state
        u_dim = 2 # dimension of the control output
        d = 0.5 # length of the robot (in Ackermann modeling)
        N = len(self.ref_traj) # number of timesteps in the reference trajectory 
        
        ## Tuning
        iterations = 100 # (max) number of iterations
        q = 10 # tuning parameter for q -> state penalty
        qf = 50 # tuning parameter for final q
        r = 1 # tunign parameter for u -> control action penalty

        ## Cost function
        Q = q * np.eye(s_dim)
        Qf = qf * np.eye(s_dim)
        R = r * np.eye(u_dim)

        ## Preallocate matrices
        traj_x_ref = np.zeros((iterations ,N, s_dim,1))
        traj_u_ref = np.zeros((iterations, N, u_dim,1))
        R_hom = np.zeros((iterations ,N, u_dim+1, u_dim+1))
        Q_hom = np.zeros((iterations ,N, s_dim+1, s_dim+1))
        P_hom = np.zeros((iterations ,N, s_dim+1, s_dim+1))
        K_hom = np.zeros((iterations ,N, u_dim+1, s_dim+1))
     
        ## Set up reference trajectory TODO Check if it stays the same all the time -> I would say yes :)
        traj_x_ref[:,:,0:2,0] = self.ref_traj # i = 0 -> anyway it should be the same for all iteraions?!
        traj_x_ref = add_theta(traj_x_ref)
   
        # Initialize algorithm - First iteration
        ## Use reference trajectory and u=0; but maybe u=PID?
        ## u
        traj_u = np.zeros((iterations ,N, u_dim,1)) # Set initial trajectory to 0 
        traj_u[:,:,:,0] =3.14/180
        ## x (state)
        traj_x = np.zeros((iterations ,N, s_dim,1))
        traj_x[0,:,0:2,0]  = self.ref_traj # Set initial trajectory to reference trajectory
        traj_x = add_theta(traj_x)        
        traj_x[:,current_timestep,:,0] = pose
        
        ### Loop over i ###
        for i in range(0, iterations-1, 1):

            # Set up trajectories

            # Quadricize cost about trajectory
            for t in range(current_timestep,N-1,1):
                ## Be aware Transpose for x, and u is a column vector       
                Q_hom[i,t,:,:] = get_Q_hom(traj_x_ref, Q, i, t)

                ## Calculate R_hom
                R_hom_12 = R @ np.array([traj_u[i,t,:,0]-traj_u_ref[i,t,:,0]]).T
                # = R(u_t^i - u_t^ref)

                R_hom_21 = np.array([traj_u[i,t,:,0]-traj_u_ref[i,t,:,0]])
                # = (u_t^i - u_t^ref)^T

                R_hom_22 = np.array([traj_u[i,t,:,0]-traj_u_ref[i,t,:,0]]) @ np.array([traj_u[i,t,:,0]-traj_u_ref[i,t,:,0]]).T
                # = (u_t^i - u_t^ref)^T * (u_t^i - u_t^ref)

                R_hom[i,t,:,:] = np.concatenate((np.vstack((R,R_hom_21)),np.vstack((R_hom_12,R_hom_22))), axis=1)
            
            P_hom[i,N-1,:,:] = get_Q_hom(traj_x_ref, Qf, i, N-1) # N-1 as we start counting with 0
            # Backward pass
            for t in range(N-2,current_timestep,-1): # N-2 as we set the last and we start counting with 0
                A_hom, B_hom = linearize_dynamics(v_i = traj_u[i,t,0,0], delta_i = traj_u[i,t,1,0], theta_i =  traj_x_ref[i,t,2,0], x_it1 = traj_x[i+1,t,0,0], y_it1 = traj_x[i+1,t,1,0], theta_it1 = traj_x[i+1,t,2,0]) # Calculate A, B and f

                #P_hom[i,t,:,:] = Q_hom[i,t,:,:] + np.matmul(np.matmul(np.transpose(K_hom[i,t,:,:]), R_h[i,t,:,:]),K_hom[i,t,:,:]) + np.matmul(np.matmul(np.transpose(A_hom + np.matmul(B_hom[i,t,:,:], K_hom[i,t,:,:])), P_hom[i,t+1,:,:]),(A_hom + A_hom + np.matmul(B_hom, K_hom[i,t,:,:])))
                f = np.matmul(np.matmul(np.transpose(K_hom[i,t,:,:]), R_hom[i,t,:,:]), K_hom[i,t,:,:])
                g = np.transpose(A_hom + np.matmul(B_hom, K_hom[i,t,:,:]) )
                l = np.transpose(g)
                P_hom[i,t,:,:] = Q_hom[i,t,:,:] + f + (g@P_hom[i,t+1,:,:]@l)

                #K_hom[i,t,:,:] = np.matmul(np.matmul(np.matmul(-np.linalg.pinv((R_h[i,t,:,:] + np.matmul(np.matmul(np.transpose(B_hom), P_hom[i,t+1,:,:]),B_hom))),np.transpose(B_hom)),P_hom[i,t+1,:,:]), A_hom)

                par_k = -np.linalg.pinv(R_hom[i,t,:,:] + np.matmul(np.matmul(np.transpose(B_hom),P_hom[i,t+1,:,:]),B_hom))
                K_hom[i,t,:,:] = np.matmul(np.matmul(np.matmul(par_k, np.transpose(B_hom)),P_hom[i,t+1,:,:]), A_hom)
            

            # Forward pass
            for t in range(1+current_timestep,N-1,1):
            # Calculate u
                #traj_u = np.zeros((iterations ,N, u_dim))  
                #traj_x = np.zeros((iterations ,N, s_dim))
                # B_ht = np.append(B_t, [[0,0]], axis=0)-----TO APPEND 
                vec_hom = np.append([[traj_x[i+1,t,0,0]-traj_x[i,t,0,0]],[traj_x[i+1,t,1,0]-traj_x[i,t,1,0]], [traj_x[i,t,2,0]-traj_x[i+1,t,2,0]]], [[1]], axis=0) #where is x_i+1 ??? --> we're assuming x_i=x_ref, x_i+1=x_i
                mul_hom = K_hom[i,t,:,:]@vec_hom
                mul =  mul_hom[0:2] #remove the last line - index 2 - from mul_hom to be consistent with u  
                traj_u[i+1,t,:,0] = traj_u[i,t,:,0] + mul.T

            # Calculate new x 
                #f_x_xt1 = np.array([[v_i*np.cos(theta_i) - x_it1], [v_i*np.sin(theta_i) - y_it1], [v_i*np.tan(delta_i)/d - theta_it1]])
                row_1 = [traj_u[i+1,t,0,0]*np.cos(traj_x[i+1,t,2,0]) + traj_x[i+1,t,0,0]]
                row_2 = [traj_u[i+1,t,0,0]*np.sin(traj_x[i+1,t,2,0]) + traj_x[i+1,t,1,0]]
                row_3 = [traj_u[i+1,t,0,0]*np.tan(traj_u[i+1,t,1,0])/d + traj_x[i+1,t,2,0]]
                traj_x[i+1,t+1,:,0] = np.concatenate((row_1,row_2,row_3),axis=0)

            # TODO Check cost -> maybe break!
            ### Loop ###
        speed = traj_u[iterations-1, current_timestep, 0,0]
        steering_angle = traj_u[iterations-1, current_timestep, 1,0]
        print("time", current_timestep)
        print("u", traj_u[iterations-1, :, :,0])
        print("x", traj_x[iterations-1, :, :,0])
        
        self.x_plot = traj_x[10-1,:,:,0] 

        return np.array([steering_angle, speed])
        #### END OF YOUR CODE ####
        raise NotImplementedError
        
    def optimal_control(self, pose):
        #### YOUR CODE HERE ####
        
        
        # return np.array([steering_angle, speed])
        #### END OF YOUR CODE ####
        raise NotImplementedError
        

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
        rclpy.logging.get_logger('lab1').info("Lap Time: " + str(tock - tick))
        print("Cross Track Error: " + str(lab1.cross_track_accumulated_error))
        print("Along Track Error: " + str(lab1.along_track_accumulated_error))
        print("Lap Time: " + str(tock - tick))

    lab1.destroy_node()
    rclpy.shutdown()
    
    
if __name__ == '__main__':
    controller_type = sys.argv[1]
    main(controller_type)
    
    
        