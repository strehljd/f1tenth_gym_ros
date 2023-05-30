import numpy as np
import matplotlib.pyplot as plt
import os


# get the reference trajectory
self_ref_traj = np.load(os.path.join('/Users/jstrehl/Documents/git/technion/f1tenth_gym_ros/', 'resource','ref_traj.npy'))

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
current_timestep = 0
# Parameters
s_dim = 3 # dimension of the state
u_dim = 2 # dimension of the control output
d = 0.5 # length of the robot (in Ackermann modeling)
N = len(self_ref_traj) # number of timesteps in the reference trajectory 

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
traj_x_ref[:,:,0:2,0] = self_ref_traj # i = 0 -> anyway it should be the same for all iteraions?!
traj_x_ref = add_theta(traj_x_ref)

# Initialize algorithm - First iteration
## Use reference trajectory and u=0; but maybe u=PID?
## u
traj_u = np.zeros((iterations ,N, u_dim,1)) # Set initial trajectory to 0 
## x (state)
traj_x = np.zeros((iterations ,N, s_dim,1))
traj_x[0,:,0:2,0]  = self_ref_traj # Set initial trajectory to reference trajectory
traj_x = add_theta(traj_x)        
traj_x[:,current_timestep,:,0] = np.array([0,0,0])

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
        A_hom, B_hom = linearize_dynamics(v_i = traj_u[i,t,0,0], delta_i = traj_u[i,t,1,0], theta_i =  traj_x[i,t,2,0], x_it1 = traj_x[i,t+1,0,0], y_it1 = traj_x[i,t+1,1,0], theta_it1 = traj_x[i,t+1,2,0]) # Calculate A, B and f

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

### Debugging output ###

print("traj_x", traj_x[iterations-1,0:100,:,0])