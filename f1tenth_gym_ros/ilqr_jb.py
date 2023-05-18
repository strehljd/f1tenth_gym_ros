import numpy as np
import os


class ilqr:
    def __init__(self):
        # working directory
        cwd = os.getcwd()
        cwd_up = os.path.dirname(cwd)

        # parameters
        self.d = 0.3302

        # tuning parameter
        self.dt = 0.5
        self.max_iter = 100
        self.Q = 20*np.eye(3)
        self.R = 2*np.eye(2)

        # reference data
        self.x_ref = np.load(os.path.join(cwd, 'resource', 'ref_traj.npy'))
        self.max_time = len(self.x_ref)
        self.u_ref = np.zeros((self.max_time,2,1))
        self.x_init = np.load(os.path.join(cwd, 'resource', 'true_pos.npy'))
        self.u_init = np.load(os.path.join(cwd, 'resource', 'controllers.npy'))

        # evolving data
        self.u_traj = np.zeros((self.max_iter, self.max_time, 2, 1))
        self.x_traj = np.zeros((self.max_iter, self.max_time, 3, 1))
        self.A = np.zeros((self.max_iter, self.max_time, 3, 3))
        self.B = np.zeros((self.max_iter, self.max_time, 3, 2))
        self.S = np.zeros((self.max_iter, self.max_time,3,3))
        self.s = np.zeros((self.max_iter, self.max_time,3,1))
        self.d_bw = np.zeros((self.max_iter, self.max_time, 2, 1))
        self.K_bw = np.zeros((self.max_iter, self.max_time, 2, 3))
        self.rho = 1

        # initialization
        self.u_traj[0, :, :, 0] = self.u_init[:self.max_time]
        self.x_traj[0, :, :, 0] = self.x_init[:self.max_time]
        #self.S =
        #self.s =

    # linearized dynamics
    def get_AB(self, i, t): 
        # Linearization based on jacobian linearization
        v_t = self.u_traj[i, t, 0]
        delta_t = self.u_traj[i, t, 1]
        theta_t = self.x_traj[i, t, 2]
        dt = self.dt
        d = self.d
        self.A[i, t] = np.array([[1, 0, -v_t *np.sin(theta_t)*dt], [0, 1, v_t * np.cos(theta_t)*dt], [0, 0, 1]])
        self.B[i, t] = np.array([[np.cos(theta_t)*dt, 0],[np.sin(theta_t)*dt, 0 ], [np.tan(delta_t)*dt/d, v_t*dt/(d*np.cos(delta_t)**2)]])
        pass    

    def compute_rho(self):
        rho = 1
        return rho
    
    # computing dt and Kt backwards
    def backwards(self, i, t):
        R = self.R
        B_t = self.B[i,t]
        A_t = self.A[i,t]
        
        ### Regularization
        rho = self.compute_rho()
        S_tp1 = self.S[i, t+1] + rho*np.eye(3)
        ###

        s_tp1 = self.s[i, t+1]
        u_t = self.u_traj[i, t]

        # update d und K
        self.d_bw[i, t] = -np.linalg.inv(R+B_t.T@S_tp1@B_t)@(u_t.T@R-2*u_t.T@R+s_tp1.T@B_t).T #### ! Transposed or not?
        self.K_bw[i, t] = -np.linalg.inv(R+B_t.T@S_tp1@B_t)@(B_t.T@S_tp1@A_t)
        pass

    # computing u and x forwards
    def forwards(self, i, t):
        u_it = self.u_traj[i, t]
        x_it = self.x_traj[i, t]
        x_itp1 = self.x_traj[i, t+1]
        x_ip1t = self.x_traj[i+1, t]
        K_t = self.K_bw[i, t]
        d_t = self.d_bw[i, t]
        dt = self.dt
        d = self.d
        
        # update control vector
        self.u_traj[i+1, t] = u_it+K_t@(x_ip1t-x_it) + d_t

        # update state
        self.x_traj[i+1, t+1, 0] = x_itp1[0] + dt*(u_it[0]*np.cos(x_it[2])*dt + x_it[0])
        self.x_traj[i+1, t+1, 1] = x_itp1[1] + dt*(u_it[0]*np.sin(x_it[2])*dt + x_it[1])
        self.x_traj[i+1, t+1, 2] = x_itp1[2] + dt*(u_it[0]*np.tan(u_it[1])*dt/d + x_it[2])
        pass

    
    def run_ilqr(self):
        # run iterations
        for i in range(self.max_iter-1):
            
            # run backwards pass
            for t in range(self.max_time-2, 0, -1):
                self.backwards(i, t)
            
            # run forwards pass
            for t in range(self.max_time-1):
                self.forwards(i, t)


          

if __name__ == '__main__':
    ilqr = ilqr()
    ilqr.run_ilqr()