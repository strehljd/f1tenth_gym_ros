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
        self.backward_pass(x_traj, x_desired, Q, R, Q_terminal)
        u_new = u_init.copy()

        for i in range(self.num_iterations):
            u_new += self.k_feedforward[i] + np.dot(self.K[i], x_traj[i] - x_init)
            x_traj = self.forward_pass(x_init, u_new)

        return u_new[0]

    def forward_pass(self, x_init, u_init):
        x_traj = np.zeros((self.num_iterations + 1, self.num_states))
        x_traj[0] = x_init

        for i in range(self.num_iterations):
            x_traj[i+1] = self.dynamics(x_traj[i], u_init[i])

        return x_traj

    def backward_pass(self, x_traj, x_desired, Q, R, Q_terminal):
        Qf = Q_terminal
        V = np.zeros((self.num_iterations + 1, self.num_states))
        V[-1] = Qf.dot(x_traj[-1] - x_desired)

        for i in range(self.num_iterations - 1, -1, -1):
            A, B = self.linearize_dynamics(x_traj[i], u_init[i])
            Qx = Q.dot(x_traj[i] - x_desired)
            Qu = R.dot(u_init[i])
            Qxx = Q + np.dot(A.T, V[i+1]).dot(A)
            Quu = R + np.dot(B.T, V[i+1]).dot(B)
            Qux = np.dot(B.T, V[i+1]).dot(A)

            Quu_inv = np.linalg.inv(Quu)
            K = -np.dot(Quu_inv, Qux)
            k_feedforward = -np.dot(Quu_inv, Qu)
            V[i] = Qx + np.dot(A.T, V[i+1] + np.dot(K.T, Quu)).dot(A)
            self.K[i] = K
            self.k_feedforward[i] = k_feedforward

    def dynamics(self, x, u):
        # Define the system dynamics function
        x_0 = x[0] + u[0] * np.cos(x[2])
        x_1 = x[1] + u[0] * np.sin(x[2])
        x_2 = x[2] + u[0] * np.tan(u[1]) *1/self.distance
        return np.array([[x_0],[x_1],[x_2]])


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
