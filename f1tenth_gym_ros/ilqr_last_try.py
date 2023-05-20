import numpy as np

class iLQRController:
    def __init__(self, dt, num_iterations, num_states, num_controls):
        self.dt = dt
        self.num_iterations = num_iterations
        self.num_states = num_states
        self.num_controls = num_controls
        self.K = np.zeros((num_iterations, num_controls, num_states))
        self.k_feedforward = np.zeros((num_iterations, num_controls))
        self.distance = 0.0381 # distance between the two wheel axles


    def compute_control_input(self, x_init, u_init, x_desired, Q, R, Q_terminal):
        x_traj = self.forward_pass(x_init, u_init)
        self.backward_pass(x_traj, x_desired, Q, R, Q_terminal, u_init)
        u_new = u_init.copy()

        for i in range(self.num_iterations):
            u_new += self.k_feedforward[i] + np.matmul(self.K[i], x_traj[i] - x_init)
            x_traj = self.forward_pass(x_init, u_new)

        return u_new[0]

    def forward_pass(self, x_init, u_init):
        x_traj = np.zeros((252, self.num_states))
        x_traj[0] = x_init

        for i in range(len(x_traj)-1):
            x_traj[i+1] = self.dynamics(x_traj[i], u_init[i])

        return x_traj

    def backward_pass(self, x_traj, x_desired, Q, R, Q_terminal, u_init):
        Qf = Q_terminal
        S = np.zeros((self.num_iterations + 1, self.num_states, self.num_states))
        s = np.zeros((self.num_iterations +1, self.num_states))
        S[-1] = Qf 
        s[-1] = Qf @(x_traj[251] - x_desired[251])
        print(S[-1])

        for i in range(self.num_iterations - 1, -1, -1):
            A, B = self.linearize_dynamics(x_traj[i], u_init[i])
            Qx = Q@(x_traj[i] - x_desired[i]) + s[i+1] # CDONE
            Qu = R@u_init[i]  + s[i+1]@B # CDONE
            
            Qxx = Q + np.matmul(A.T, S[i+1])@(A) # CDONE
            Quu = R + (B.T @ S[i+1] @ B) # CDONE

            Qux = np.matmul(B.T, S[i+1])@(A) # CDONE

            Quu_inv = np.linalg.inv(Quu) # CDONE
            K = -np.matmul(Quu_inv, Qux) # CDONE

            d_t = -np.matmul(Quu_inv, Qu) # CDONE
            S[i] = Qxx + K.T@Quu@K + K.T@Qux + Qux.T@K # CDONE
            s[i] = Qx + K.T@Quu@d_t + K.T@Qu + Qux.T@d_t
            
            self.K[i] = K
            self.k_feedforward[i] = d_t

    def dynamics(self, x, u):
        # Define the system dynamics function
        x_0 = x[0] + u[0] * np.cos(x[2])
        x_1 = x[1] + u[0] * np.sin(x[2])
        x_2 = x[2] + u[0] * np.tan(u[1]) *1/self.distance
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
    # x_init = np.load("/Users/jstrehl/Documents/git/technion/f1tenth_gym_ros/resource/x_init.npy")
    x_init = np.array([0, 0, 0])
    u_init = np.load("/Users/jstrehl/Documents/git/technion/f1tenth_gym_ros/resource/u_init.npy")
    x_desired = np.zeros([3,252])
    x_desired[0:2,:] = np.transpose(np.load("/Users/jstrehl/Documents/git/technion/f1tenth_gym_ros/resource/ref_traj.npy"))
    print("Trajectories loaded!")

    for j in range(len(x_desired)-1):
        x_desired[2,j] =  np.arctan2(x_desired[1,j+1]-x_desired[1,j], x_desired[0,j+1]-x_desired[0,j])

    controller = iLQRController(0.5,100, 3, 2) # instatiate a controller
    controller.compute_control_input(x_init, u_init, np.transpose(x_desired), Q, R, Q_terminal)
    print("Iterations done")

if __name__ == "__main__":
    main()