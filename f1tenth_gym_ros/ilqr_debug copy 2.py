import numpy as np
import matplotlib.pyplot as plt
import os

# get the reference trajectory
self_ref_traj = np.load(os.path.join('/Users/jstrehl/Documents/git/technion/f1tenth_gym_ros/', 'resource','ref_traj.npy'))


dt = 0.5

# FUNCTIONS ###############################################################################################
def linearize_dynamics(x_ti ,y_ti ,theta_ti, delta_ti, v_ti, dt, d, x_t1i, y_t1i, theta_t1i, v_t1i, delta_t1i, switch):
    # state, which the lineariztion is made "around"
    # x_t^i = x_ti
    # y_t^i = y_ti
    # theta_t^i = theta_ti
    # --
    # delta_t^i = delta_ti
    # v_t^i = v_ti 

    # Fixed parames
    # delta_t (timestep length) = dt
    # wheel lenght of the robot = d

    # State at (t+1)
    # x_{t+1}^i = x_t1i
    # y_{t+1}^i = y_t1i
    # theta_{t+1}^i = theta_t1i 

    if switch: 
        # Linearization as discused with Harrel
        # Calculate the linearized state-space equation based on the current u and state

        # Linearization based on jacobian linearization
        A_t = np.eye(3)
        B_t = np.array([ [ dt*np.cos(theta_ti) , 0 ],
                        [ dt*np.sin(theta_ti) , 0 ],
                        [ 0,         v_ti * (1/d) * dt * (1/(np.square(np.cos(delta_ti))))         ]  ])
        f_x_xt1 = np.array([ [v_ti*np.cos(theta_ti) - x_t1i      ],
                            [v_ti*np.sin(theta_ti) - y_t1i      ],
                            [v_ti*np.tan(delta_ti)/d - theta_t1i]])

        # Represent in homogenous coordinate systems
        A_ht = np.append(np.concatenate((A_t,f_x_xt1), axis=1), [[0,0,0,1]], axis=0)
        # print("A_ht: {}".format(A_ht))
        B_ht = np.append(np.concatenate((B_t, np.array([[0,0,0]]).T), axis=1), [[0,0,1]], axis=0)
    else:    
        #Linearization as proposed by jan
        # Define the matriced based on a Taylor-Approximation
        A_t = np.array([[1, 0, -v_ti * np.sin(theta_ti) * dt],
                        [0, 1, v_ti * np.cos(theta_ti) * dt],
                        [0, 0, 1]])
        
        B_t = np.array([[np.cos(theta_ti) * dt, 0],
                        [np.sin(theta_ti) * dt, 0 ],
                        [0, v_ti * (1/d) * dt * (1/(np.square(np.cos(delta_ti)))) ]])
        
        f_x_xt1 = np.array([[v_ti*np.cos(theta_ti) * dt + x_ti- x_t1i], 
                        [v_ti*np.sin(theta_ti) * dt + y_ti - y_t1i], 
                        [v_ti*np.tan(delta_ti)/d * dt + theta_ti - theta_t1i]])

        # Represent in homogenous coordinate systems
        A_ht = np.append(np.concatenate((A_t,f_x_xt1), axis=1), [[0,0,0,1]], axis=0)
        #print("A_ht: {}".format(A_ht))
        B_ht = np.append(np.concatenate((B_t, np.array([[0,0,0]]).T), axis=1), [[0,0,1]], axis=0)

    return A_ht, B_ht  

def add_theta(traj_x, ref=False):
    # Calculate theta_ref based on the tangent between the current and the next waypoint
    for j in range(N):
        if ref == True:
            traj_x[j,2] =  np.arctan2(traj_x[(j+1)%N,1]-traj_x[j%N,1], traj_x[(j+1)%N,0]-traj_x[j%N,0])
        else:
            traj_x[0,j,2] =  np.arctan2(traj_x[0,(j+1)%N,1]-traj_x[0,j%N,1], traj_x[0,(j+1)%N,0]-traj_x[0,j%N,0])
    return traj_x       

def get_Q_hom(traj_x, traj_x_ref, Q, i, t):
    x_diff = np.array([traj_x[i,t,:,0]-traj_x_ref[t,:,0]]).T
    Q_hom_12 = Q @ x_diff
    Q_hom_21 = x_diff.T @ Q
    Q_hom_22 = x_diff.T @ x_diff     
    return np.concatenate((np.vstack((Q,Q_hom_21)),np.vstack((Q_hom_12,Q_hom_22))), axis=1)

def get_R_hom(traj_u, traj_u_ref, R, i, t):
    u_diff = np.array([traj_u[i,t,:,0]-traj_u_ref[t,:,0]]).T
    R_hom_12 = R @ u_diff
    R_hom_21 = u_diff.T
    R_hom_22 = u_diff.T @ u_diff
    return np.concatenate((np.vstack((R,R_hom_21)),np.vstack((R_hom_12,R_hom_22))), axis=1)

def get_cost(traj_x, traj_x_ref, traj_u, traj_u_ref, Q_hom, R_hom, i, t):
    x_diff = np.append(np.array([traj_x[i,t,:,0]-traj_x_ref[t,:,0]]).T, [[1]], axis=0)
    u_diff = np.append(np.array([traj_u[i,t,:,0]-traj_u_ref[t,:,0]]).T, [[1]], axis=0)
    cost = x_diff.T @ Q_hom[i][t] @ x_diff + u_diff.T @ R_hom[i][t] @ u_diff
    if t ==(N-1):
        cost = x_diff.T @ Q_hom[i][t] @ x_diff # in the final iteration we dont have a control input. Thus, cost is only the state cost.
    return cost


current_timestep = 0
# self.get_ref_pos()

# Parameters
s_dim = 3 # dimension of the state
u_dim = 2 # dimension of the control output
d = 0.3302 # length of the robot (in Ackermann modeling)
N = len(self_ref_traj) # number of timesteps in the reference trajectory 

#   #   #   #   #   #   #   #   #   #   #   #   #   #   #
## Tuning
max_iterations = 10 # (max) number of max_iterations
q = 1 # tuning parameter for q -> state penalty
qf = 1 # tuning parameter for final q
r = 1.5 # tunign parameter for u -> control action penalty
cost_criteria = 1
#   #   #   #   #   #   #   #   #   #   #   #   #   #   #

## Cost function
Q = q * np.array([[1,0,0],[0,1,0],[0,0,3]])
Qf = qf * np.array([[1,0,0],[0,1,0],[0,0,3]])
R = r * np.array([[2,0],[0,1]])

## Preallocate matrices
traj_x_ref = np.empty((N, s_dim, 1))
traj_x_ref[:] = np.nan
traj_u_ref = np.empty((N, u_dim, 1))
traj_u_ref[:] = np.nan
R_hom = np.empty((max_iterations ,N, u_dim+1, u_dim+1))
R_hom[:] = np.nan
Q_hom = np.empty((max_iterations ,N, s_dim+1, s_dim+1))
Q_hom[:] = np.nan
P_hom = np.empty((max_iterations ,N, s_dim+1, s_dim+1))
P_hom[:] = np.nan
K_hom = np.empty((max_iterations ,N, u_dim+1, s_dim+1))
K_hom[:] = np.nan

## Set up reference trajectory TODO Check if it stays the same all the time -> I would say yes :)
traj_x_ref[:,0:2,0] = self_ref_traj 
traj_x_ref = add_theta(traj_x_ref, ref=True)

# Initialize algorithm - First iteration
traj_u = np.empty((max_iterations ,N, u_dim,1)) # Set initial trajectory to 0
traj_u[:] = np.nan 

# Guess control inputs for the first iteration
traj_u[0,:,:,0] = 0 # speed for all timestamps
# Set angle with 1s turn for (hopefully) 90 [deg]
# traj_u[0,0:38,1,0] = 0
# traj_u[0,38:40,1,0] = np.arctan(3.14*d/(0.5))
# traj_u[0,40:72,1,0] = 0
# traj_u[0,72:74,1,0] = np.arctan(3.14*d/(0.5))
# traj_u[0,74:163,1,0] = 0
# traj_u[0,163:165,1,0] = np.arctan(3.14*d/(0.5))
# traj_u[0,165:197,1,0] = 0
# traj_u[0,197:251,1,0] = np.arctan(3.14*d/(0.5))

traj_u[:,N-1,:,0] = 0


traj_u_ref[:] = 0


## x (state)
traj_x = np.empty((max_iterations ,N, s_dim,1))
traj_x[:] = np.nan 
traj_x[:,0,:,0] = traj_x_ref[0,:,0] # Set each initial position in each iteration to start position
traj_x[0,:,:,0] = traj_x_ref[:,:,0]        
#traj_x = add_theta(traj_x)

####################################################
################## Iteration Loop ##################
####################################################

runs = 0
costs = np.empty((max_iterations, N))
costs[:] = np.nan 
cost_sum = np.empty((max_iterations, 1))
cost_sum[:] = np.nan 

for i in range(max_iterations-1):

    # Calculate Q and R
    for t in range(N):
        ## Be aware Transpose for x, and u is a column vector       
        Q_hom[i,t,:,:] = get_Q_hom(traj_x, traj_x_ref, Q, i, t)
        R_hom[i,t,:,:] = get_R_hom(traj_u, traj_u_ref, R, i, t)
    
    # Final Cost
    P_hom[i,N-1,:,:] = get_Q_hom(traj_x, traj_x_ref, Qf, i, N-1) # N-1 as we start counting with 0
    
    
    ####### Backward pass

    for t in range(N-2,current_timestep-1,-1): # N-2 as we set the last and we start counting with 0
        A_bw, B_bw = linearize_dynamics(traj_x[i,t,0,0], traj_x[i,t,1,0], traj_x[i,t,2,0], traj_u[i, t, 1,0], traj_u[i, t, 0,0], dt, d, traj_x[i,t+1,0,0], traj_x[i,t+1,1,0], traj_x[i,t+1,2,0], "Hello world" ) # Calculate A, B and f
        Q_bw = Q_hom[i,t,:,:]
        R_bw = R_hom[i,t,:,:]
        P_bwt1 = P_hom[i,t+1,:,:]
 
        K_hom[i,t,:,:] = -1*(np.linalg.pinv(R_bw + (B_bw.T @ P_bwt1 @B_bw)) @ B_bw.T @ P_bwt1 @ A_bw)

        K_bw = K_hom[i,t,:,:]
        P_hom[i,t,:,:] = Q_bw + K_bw.T @ R_bw @ K_bw + (A_bw+ (B_bw @ K_bw)).T @ P_bwt1 @ (A_bw + (B_bw @ K_bw))
    
    ####### Forward pass
    for t in range(current_timestep,N-1,1):
    # Calculate new u
        u_fp = traj_u[i,t,:,0]
        K_fp = K_hom[i,t,:,:]
        x_diff = np.append(np.array([traj_x[i+1,t,:,0]-traj_x[i,t,:,0]]).T, [[1]], axis=0)
        traj_u[i+1,t,:,0] = u_fp + (K_fp @ x_diff)[:2,0]

    # Calculate new x 
    # Why did u use dt here? This should be the nonlinear model! -> removed dt and changed +xi instead of +xi+1
        x_new = np.array([[traj_u[i+1,t,0,0]*np.cos(traj_x[i+1,t,2,0]) + traj_x[i+1,t,0,0]],
                            [traj_u[i+1,t,0,0]*np.sin(traj_x[i+1,t,2,0]) + traj_x[i+1,t,1,0]],
                            [traj_u[i+1,t,0,0]*np.tan(traj_u[i+1,t,1,0])/d + traj_x[i+1,t,2,0]]])
        traj_x[i+1,t+1,:,:] = x_new
        
    for t in range(N):
        # append current cost in timestep, iteration
        costs[i, t] = get_cost(traj_x, traj_x_ref, traj_u, traj_u_ref, Q_hom, R_hom, i, t)
    
    cost_sum[i] = (np.sum(costs[i]))

    # stopping criteria
    if (cost_sum[i] <= cost_criteria) and (i>0):
        break
    runs += 1

    # Add current trajectory to plot
    plt.plot(traj_x[i,:,0,0],traj_x[i,:,1,0],label=str(i))

### end of iteration loop





#### END OF YOUR CODE ####