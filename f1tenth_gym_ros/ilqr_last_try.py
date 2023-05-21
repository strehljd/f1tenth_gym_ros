import numpy as np

class iLQRController:
    def __init__(self, dt, num_iterations, num_states, num_controls, num_timesteps):
        self.dt = dt
        self.num_iterations = num_iterations
        self.num_timesteps = num_timesteps
        self.num_states = num_states
        self.num_controls = num_controls
        self.K = np.zeros((num_timesteps, num_controls,num_states))
        self.k = np.zeros((num_timesteps, num_controls))
        self.d_t = np.zeros((num_timesteps, num_controls))
        self.distance = 0.0381 # distance between the two wheel axles TODO move to init
        self.cost = 0 # cost of the last iteration

    def compute_control_input(self, x_init, u_init, x_reference, Q, R, Q_terminal):
        # First iteration based on initial values
        x = x_init
        u = u_init

        rho = 0.1 # set to starting value

        # Calculate iterations
        count = 0
        while count < self.num_iterations:
            cost,x,u = self.compute_iteration(x, u, x_reference, Q, R, Q_terminal,x_init, rho)

            # Calculate regularization parameter based on explination on the blackboard in the last tutorial
            if cost < self.cost:
                rho = rho * 0.9
            else:
                rho = rho * 1.1
            
            self.cost = cost # store cost unitl next iteration

            count+=1
        pass

    def get_cost(self, x, u, x_reference, Q, Q_terminal, R):
        # Calculate cost according to equation (1) in the lecture's tutorial slide 13
        c = 0
        for t in range(self.num_timesteps):
            c += (x[t] - x_reference[t]).T @ Q @ (x[t] - x_reference[t]) # state cost
            c += (u[t]).T @ R @ (u[t]) # control cost with the assumption that u_ref[for all t] = 0
        return c
    
    def compute_iteration(self, x, u, x_reference, Q, R, Q_terminal,x_init, rho):

        # Calculate new K_t and d_T
        self.backward_pass(x, x_reference, Q, R, Q_terminal, u, rho)

        # Calculate new trajecories
        x, u = self.forward_pass(x, u,x_init)

        cost = self.get_cost(x, u, x_reference, Q, Q_terminal, R)
        
        return (cost, x, u)

    def forward_pass_old(self, x, u, x_init):
        # x, u are the trajectories of the current iteration (i)
        x_traj = np.zeros((self.num_timesteps, self.num_states)) # trajectory in the next iteration (i+1)
        x_traj[0] = x_init[0] #set initial value to initial value at the first iteration

        u_traj = np.zeros((self.num_timesteps, self.num_controls)) # control in the next iteration (i+1)

        for t in range(self.num_timesteps-1):
            u_traj[t] = u[t] + self.K[t]@(x_traj[t] - x[t]) + self.d_t[t] # here we could add an alpha
            x_traj[t+1] = self.dynamics(x_traj[t], u_traj[t])

        return (x_traj, u_traj)
    
    def forward_pass(self, x, u, x_init):
        # Forward pass according to section 8.2 in Calinon's paper
        # x, u are the trajectories of the current iteration (i)
        x_traj = np.zeros((self.num_timesteps, self.num_states)) # trajectory in the next iteration (i+1)
        x_traj[0] = x_init[0] #set initial value to initial value at the first iteration

        u_traj = np.zeros((self.num_timesteps, self.num_controls)) # control in the next iteration (i+1)

        for t in range(self.num_timesteps-1):
            u_traj[t] = self.k[t] + self.K[t] @ (x_traj[t] - x[t]) + u[t]

            x_traj[t+1] = self.dynamics(x_traj[t], u_traj[t])

        return (x_traj, u_traj)
    def backward_pass(self, x_traj, x_reference, Q, R, Q_terminal, u_traj, rho):
        # Backward pass according to section 8.2 in Calinon's paper
        
        vx = np.zeros((self.num_timesteps +1, self.num_states))
        Vxx = np.zeros((self.num_timesteps + 1, self.num_states, self.num_states))

        Vxx[-1] = Q_terminal #euqual to S
        vx[-1] = Q_terminal @(x_traj[self.num_timesteps-1] - x_reference[self.num_timesteps-1]) # equal to s


        for t in range(self.num_timesteps-1):
            A, B = self.linearize_dynamics(x_traj[t], u_traj[t])


            # Based on the cost equation on slide 17 in the lectures tutorial it can be derived, that
            gu = u_traj[t].T  + vx[t+1] @ B
            gx = x_reference[t].T @ Q - 2*x_reference[t].T @ Q + vx[t+1].T @ A
            Huu = R
            Hxx = 0
            Hux = 0
            qx = gx + A.T @ vx[t+1]
            qu = gu + B.T @ vx[t+1]
            Qxx = Hxx + A.T @ Vxx[t+1] @ A
            Quu = Huu + B.T @ Vxx[t+1] @ B
            Qux = Hux + B.T @ Vxx[t+1] @ A   
            Quu_inv = np.linalg.inv(Quu)
            
            # Calculate regularized Quu and Qux
            Quu_tilde = Huu + (B.T @ (Vxx[t+1]+ rho*np.eye(self.num_states)) @ B)
            Qux_tilde = B.T @ (Vxx[t+1]+ rho*np.eye(self.num_states)) @ A
            Quu_inv_tilde = np.linalg.inv(Quu_tilde)

            vx[t] = qx - Qux.T @ Quu_inv @ qu
            Vxx[t] = Qxx - Qux.T @ Quu_inv @ Qux

            self.K[t] = -Quu_inv_tilde @ Qux_tilde
            self.k[t] = -Quu_inv_tilde @ qu

        pass 

    def backward_pass_old(self, x_traj, x_reference, Q, R, Q_terminal, u_traj):
        Qf = Q_terminal
        S = np.zeros((self.num_timesteps + 1, self.num_states, self.num_states))
        s = np.zeros((self.num_timesteps +1, self.num_states))
        S[-1] = Qf 
        s[-1] = Qf @(x_traj[self.num_timesteps-1] - x_reference[self.num_timesteps-1])

        for i in range(self.num_timesteps - 1, -1, -1):
            A, B = self.linearize_dynamics(x_traj[i], u_traj[i])

            Qx = Q@(x_traj[i] - x_reference[i]) + s[i+1]@A # CDONE
            Qu = R@u_traj[i]  + s[i+1]@B # CDONE
            
            Qxx = Q + A.T@S[i+1]@A # CDONE
            Quu = R + (B.T @ S[i+1] @ B) # CDONE
            Qux = B.T@S[i+1]@(A) # CDONE


            # Calculate regularized Quu and Qux
            rho = 0.1 # TODO add a meaningfull calculation here -> maybe based on the cost?
            Quu_tilde = R + (B.T @ (S[i+1]+ rho*np.eye(self.num_states)) @ B)
            Qux_tilde = B.T @ (S[i+1]+ rho*np.eye(self.num_states)) @ A
            Quu_inv_tilde = np.linalg.inv(Quu_tilde) # CDONE


            K = -np.matmul(Quu_inv_tilde, Qux_tilde) # CDONE
            d_t = -np.matmul(Quu_inv_tilde, Qu) # CDONE


            S[i] = Qxx + K.T.dot(Quu).dot(K) + K.T.dot(Qux) + Qux.T.dot(K) # CDONE
            s[i] = Qx + K.T@Quu@d_t + K.T@Qu + Qux.T@d_t
            
            self.K[i] = K
            self.d_t[i] = d_t
            if i == 0:
                print("")

    def dynamics(self, x, u): # Testing done
        # Define the system dynamics function
        x_0 = x[0] + u[0] * np.cos(x[2])
        x_1 = x[1] + u[0] * np.sin(x[2])
        x_2 = x[2] + u[0] * np.tan(u[1]) * 1/self.distance
        return np.array([x_0, x_1, x_2])


    def linearize_dynamics(self, x, u):
        # Linearize the system dynamics around the given state and control inputs
        A = np.array([
            [1, 0, -u[0] * np.sin(x[2])], 
            [0, 1,  u[0] * np.cos(x[2])],
            [0, 0, 1]])
        
        B = np.array([
            [np.cos(x[2]), 0], 
            [np.sin(x[2]), 0],
            [1/self.distance * np.tan(u[1]), u[0] * 1/self.distance * 1/(np.square(np.cos(u[1])))]])

        return (A, B)



def main():
    print("Starting main")

    # Define parameters
    Q = np.diag([1, 1, 1]) 
    R = np.diag([0.1, 0.1]) 
    Q_terminal = np.diag([1, 1, 1]) 

    # Load trajectories 
    x_init = np.load("/Users/jstrehl/Documents/git/technion/f1tenth_gym_ros/resource/x_init.npy")
    u_init = np.load("/Users/jstrehl/Documents/git/technion/f1tenth_gym_ros/resource/u_init.npy")
    x_reference = np.zeros([252,3])
    x_reference[:,:2] = np.load("/Users/jstrehl/Documents/git/technion/f1tenth_gym_ros/resource/ref_traj.npy")
    print("Trajectories loaded!")

    for j in range(len(x_reference)-1):
        x_reference[j,2] =  np.arctan2(x_reference[j+1,1]-x_reference[j,1], x_reference[j+1,0]-x_reference[j,0])

    controller = iLQRController(0.5,100, 3, 2, 252) # instatiate a controller

    controller.compute_control_input(x_init, u_init, x_reference, Q, R, Q_terminal)
    print("Iterations done")

if __name__ == "__main__":
    main()