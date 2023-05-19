# %%
import numpy as np
import os
import matplotlib.pyplot as plt

#-------------------------------------------------------------------
class iLQR:
    def __init__(self,controllers,true_pos,ref_traj,Q,A,B,R,dt=0.5,wheelbase = 0.3032,N=252,iter =3):
    # Trajectories and initial controllers:
        self.init_controllers =self.u  =  controllers
        self.init_pos =self.x = true_pos
        self.ref_traj= ref_traj
        self.iter = iter
        self.T = N
    # initial Matrices in a list of matrices in length T  
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.K = [np.zeros((2,3)) for _ in range(N)]
        self.S = [np.zeros((3,3)) for _ in range(N)]
        self.S[N-1] = Q
    # initial vectors:
        self.du = [np.zeros(2) for _ in range(N)]
        self.dx = [np.zeros(3) for _ in range(N)]
        self.s = [np.zeros(3) for _ in range(N)]
        self.d = [np.zeros(2) for _ in range(N)]
        self.next_pos = np.zeros(3)
        self.c = [0 for _ in range(N)]
    # initial variables:
        self.tot_c = 0
        self.dt = dt
        self.wheelbase = wheelbase
        self.rho = 1

    # Calculate one step of the cost
    def CalcCost(self,x,u,Q,R):
      return (x.T @ Q @ x + u.T @ R @ u)
    
    # Calculate the total lost
    def CalcFullCost(self,T):
        self.c = 0
        for t in range(T):
            self.c += self.CalcCost(self.x[t],self.u[t],self.Q,self.R)
    # calculate value of model f(x,u)
    def CalcModel(self,pose,control):
        x = control[1] * np.cos(pose[2])
        y = control[1] * np.sin(pose[2]) 
        theta = control[1] * np.tan(control[0]) / self.wheelbase
        return np.array([x,y,theta])
        
    # calculate the delta control
    def GetDeltaControl(self,t):
       self.du[t] = self.K[t]@(self.dx[t])+self.d[t]

    # calculate the delta of next pose
    def GetDeltaPose(self,t):
        pose = self.x[t]
        self.dx[t] = self.x[t+1] - pose

    # update pose
    def UpdatePose(self,t):
        modelVal = self.CalcModel(self.x[t+1],self.u[t+1])
        self.x[t+1] -= (self.dt * modelVal)
    
    # update the control
    def UpdateControl(self,t):
        self.u[t] -= self.du[t]

    def CalcS(self,t):
        self.S[t] = self.A[t].T @ self.S[t+1] @(self.A[t] - self.B[t]@ self.K[t]) + self.Q
        self.S[t] += self.rho * np.eye(3)

    # calculate current s:
    def calculateSmallS(self,t):
        self.s[t] = self.A[t].T @ self.S[t+1] @ self.dx[t] + self.Q @ self.dx[t]
    # calculate current d:
    def Calcd(self,t):
        inv_prod =  -np.linalg.inv(self.R+self.B[t].T @ self.S[t+1] @ self.B[t])
        self.d[t]= inv_prod @ (self.u[t].T @ self.R - 2 * self.u[t].T @ self.R + self.s[t+1].T @  self.B[t])
    
    # calculate current d:
    def CalcK(self,t):
        inv_prod = -np.linalg.inv(self.R+self.B[t].T @ self.S[t+1] @self.B[t])
        self.K[t] = inv_prod @ (self.B[t].T @ self.S[t+1]@self.A[t])

    # update A:
    def UpdateA(self,t,x,u):
        self.A[t] = np.array( [ [0,0,-u[0]*np.sin(x[2])  ],
                                [0,0,  u[0]*np.cos(x[2]) ],
                                [0,           0,       0 ] ])
    #update B:
    def UpdateB(self,t,x,u):
        self.B[t] = np.array([[np.cos(x[2]) ,                            0              ],
                              [np.sin(x[2]) ,                            0              ],
                              [np.tan(u[1])/self.wheelbase , u[0] * (1 / (np.cos(u[1])**2))    ] ] )
        
    # backward iteration
    def BackwardIter(self,t):
        self.CalcS(t)
        self.CalcK(t)
        self.calculateSmallS(t)
        self.Calcd(t)
        
        

    # forward iteration
    def ForwardIter(self,t):
        self.GetDeltaControl(t)
        self.UpdateControl(t)
        self.GetDeltaPose(t)
        self.UpdatePose(t)
        self.UpdateA(t,self.x[t],self.u[t])
        self.UpdateB(t,self.x[t],self.u[t])
        

    # iterations combined
    def FullIter(self):
        for t in reversed(range(self.T-1)):
           self.BackwardIter(t) 

        for t in range(self.T):
            self.ForwardIter(t)
        self.CalcFullCost(t)
        current_c = np.sum(self.c)
        if self.tot_c > current_c:
            self.rho *= 0.9
        else: self.rho *= 1.1
        self.tot_c = np.sum(self.c)
    # final act
    def run(self):
        for ix in range(self.iter):
            self.FullIter()
            print(f"The cost in iter. {ix} is {self.c}")

#--------------------------------------------------------
def plot_trajectory(coordinates,map = 'viridis'):
    num_points = len(coordinates)
    cmap = plt.get_cmap(map)
    colors = [cmap(i / num_points) for i in range(num_points)]

    x = [coord[0] for coord in coordinates]
    y = [coord[1] for coord in coordinates]

    plt.scatter(x, y, c=colors)
    plt.plot(x, y, color='black')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Trajectory')
    plt.colorbar(label='Time')

    # plt.show()
#--------------------------------------------------------
cwd = os.getcwd()
cwd_up = os.path.dirname(cwd)
#true_pos = np.load(cwd, 'f1tenth_gym_ros/f1tenth_gym_ros/true_pos.npy')
true_pos = np.load(cwd, 'resource', 'true_pos.npy')
controllers = np.load(cwd, 'resource', 'controllers.npy')
ref_traj = np.load(cwd,  'resource', 'ref_traj.npy')
pos = np.copy(true_pos)

# %%

wheelbase = 0.3032
N = 252
A = [np.eye(3) for _ in range(N)]
B = [np.ones((3,2)) for _ in range(N)]
Q = 1* np.eye(3)
# R = 20* np.array([[1,0],[0,10]])
R = 1e3*np.eye(2)
u = [0,0]
x = [0,0,0]

max_iter = 25
ilqr = iLQR(controllers,true_pos,ref_traj,Q,A,B,R,iter=max_iter)

# %%
ilqr.run()
plt.figure(1)
plot_trajectory(ilqr.x)
# plt.figure(2)
plot_trajectory(pos,map='inferno')
plt.show()





    
        

        



