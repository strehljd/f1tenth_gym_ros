# Debug
i = 1

# Init
##Params
N = 252 # Length of trajectory
s_dim = 3 # Dimension of the state-space
u_dim = 2 # Dimension of the control-space
d = 0.0186 # Wheel-Length
iterations = 1

## Tuning parameters
Q_linear = [1 0 0; 0 1 0; 0 0 1]
R_linear = [1 0; 0 1]

## Preallocate size
P = zeros(iterations,N,s_dim+1,s_dim+1);
K = zeros(iterations,N,u_dim+1,u_dim+1);
x = zeros(iterations,N,s_dim);
x_ref = zeros(N,s_dim);
# u[i,t,1] = speed, u[i,t,2] = steering angle
u = zeros(iterations,N,u_dim);
u_ref = zeros(N,u_dim);
Q = zeros(iterations,N,s_dim+1,s_dim+1)
R = zeros(iterations,N,u_dim+1,u_dim+1);
A = zeros(iterations,N,s_dim+1,s_dim+1);
B = zeros(iterations,N,s_dim+1,u_dim+1);


# Linearize dynamics

for t in 1:1:(N-1)
    A_left = [1 0 -u[i,t,1]*sin(x[i,t,3]); 0 1 u[i,t,1]*cos(x[i,t,3]); 0 0 1; 0 0 0] 
    A_right = [(x[i,t,1] + u[i,t,1] * cos(x[i,t,3])) - x[i,t+1,1]; (x[i,t,2] + u[i,t,1] * sin(x[i,t,3]))- x[i,t+1,2]; (x[i,t,3] + u[i,t,1] /d  * tan(u[i,t,2]))- x[i,t+1,3] ;1]
    A[i,t,:,:] = hcat(A_left, A_right)

    B[i,t,:,:] = [cos(x[i,t,3]) 0 0; sin(x[i,t,3]) 0 0; tan(u[i,t,2])/d u[i,t,1]/(d*cos(u[i,t,2]^2)) 0; 0 0 1]
end


# Quadricice cost about trajectory
for t in 1:1:(N-1)
    ## Calculate Q and R
    Q_left = vcat(Q_linear, transpose(x[i,t,:] - x_ref[t,:]))
    Q_right = vcat(Q_linear * (x[i,t,:] - x_ref[t,:]), transpose(x[i,t,:] - x_ref[t,:]) * (x[i,t,:] - x_ref[t,:]))
    Q[i,t,:,:] = hcat(Q_left, Q_right)

    R_left = vcat(R_linear, transpose(u[i,t,:] - u_ref[t,:]))
    R_right = vcat(R_linear * (u[i,t,:] - u_ref[t,:]), transpose(u[i,t,:] - u_ref[t,:]) * (u[i,t,:] - u_ref[t,:]))
    R[i,t,:,:] = hcat(R_left, R_right)

end


# LQR
## Backward Pass
P[i,N,:,:] = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]; # Is the 1 at the end right?

for t in (N-1):-1:1  
    K[i,t,:,:] = (R[i,t,:,:] + transpose(B[i,t,:,:]) * P[i,t+1,:,:] * B[i,t,:,:])^-1 * transpose(B[i,t,:,:]) * P[i,t+1,:,:] * A[i,t,:,:]
    P[i,t,:,:] = Q[i,t,:,:] + transpose(K[i,t,:,:]) * R[i,t,:,:]* K[i,t,:,:] + transpose(A[i,t,:,:] - B[i,t,:,:] * K[i,t,:,:]) * P[t+1,:,:] *(A[i,t,:,:] - B[i,t,:,:]*K[i,t,:,:]);
end

## Forward Pass
x[i,1,:] = [0 0 0]; 

for t in 1:1:(N-1)
    u[i+1,t,:] = u[i,t,:] - K[t,:,:]*  cat((x[i+1,t,:] - x[i,t,:]),[1]);
    print(cat((x[i+1,t,:] - x[i,t,:]),[1]))

    # Forward Dynamics based on the non-linear model
    x[i+1,t+1,:]  = [(x[i+1,t,1] + u[i+1,t,1] * cos(x[i+1,t,3])) (x[i+1,t,2] + u[i+1,t,1] * sin(x[i+1,t,3])) (x[i+1,t,3] + u[i+1,t,1] /d  * tan(u[i+1,t,2]))]
end