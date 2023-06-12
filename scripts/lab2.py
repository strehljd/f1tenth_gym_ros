import numpy as np
from scipy.spatial import KDTree
from PIL import Image
import yaml 
import os
import pathlib
from a_star import A_star

from matplotlib import pyplot as plt

# Plotting a graph (consisting of nodes and edges) and goal points
def plot(graph,mid_points):

    #Plot Nodes
    for point in graph['nodes']:
        plt.plot(point[0],point[1],marker="o", markersize=5, color="green")

    # Plot Edges
    for edge in graph['edges']:
        x1 = graph['nodes'][edge[0]][0]
        x2 = graph['nodes'][edge[1]][0]
        y1 = graph['nodes'][edge[0]][1]
        y2 = graph['nodes'][edge[1]][1]
        plt.plot([x1,x2],[y1,y2],'k-')

    # Plot mid points
    for goal_point in mid_points:
        plt.plot(goal_point[0],goal_point[1],marker="o", markersize=5, color="red", label="start")
    
    plt.show()


def load_map_and_metadata(map_file):
    # load the map from the map_file
    map_img = Image.open(map_file).transpose(Image.FLIP_TOP_BOTTOM)
    map_arr = np.array(map_img, dtype=np.uint8)
    map_arr[map_arr < 220] = 1
    map_arr[map_arr >= 220] = 0
    map_arr = map_arr.astype(bool)
    map_height = map_arr.shape[0]
    map_width = map_arr.shape[1]
    # TODO: load the map dimentions and resolution from yaml file
    with open(map_file.replace('.png', '.yaml'), 'r') as f:
        try:
            map_metadata = yaml.safe_load(f)
            map_resolution = map_metadata['resolution']
            map_origin = map_metadata['origin']
        except yaml.YAMLError as exc:
            print(exc)
    origin_x = map_origin[0]
    origin_y = map_origin[1]
        
    return map_arr, map_height, map_width, map_resolution, origin_x, origin_y


def pose2map_coordinates(map_resolution, origin_x, origin_y, x, y):
    x_map = int((x - origin_x) / map_resolution)
    y_map = int((y - origin_y) / map_resolution)
    return y_map, x_map

def map2pose_coordinates(map_resolution, origin_x, origin_y, x_map, y_map):
    x = x_map * map_resolution + origin_x
    y = y_map * map_resolution + origin_y
    return x, y

def collision_check(map_arr, map_height, map_width, map_resolution, origin_x, origin_y, x, y, theta):
    ####### your code goes here ####### 
    # offset computation: to take into account the position where the pose of the car is taken: not in the middle. 
    offset_x = 0.0381/2
    x = x + np.cos(theta)*offset_x 
    y = y + np.sin(theta)*offset_x 

    # TODO: transform configuration to workspace bounding box
    half_width = 1/2*np.sqrt(0.3302**2+0.2032**2) #to not take into account theta: hyp of tri
    x_min = x - half_width
    x_max = x + half_width
    half_height = 1/2*np.sqrt(0.3302**2+0.2032**2)
    y_min = y - half_height 
    y_max = y + half_height 

    # TODO: overlay workspace bounding box on map (creating borders for collision search in the next step)
    x_min_map, y_min_map = pose2map_coordinates(map_resolution, origin_x, origin_y, x_min, y_min)
    x_max_map, y_max_map = pose2map_coordinates(map_resolution, origin_x, origin_y, x_max, y_max)

    # TODO: check for collisions by looking inside the bounding box on the map if there are values greater than 0
    for i in range(x_min_map, x_max_map):
        for j in range(y_min_map, y_max_map):
            if map_arr[i,j]==1:
                return True #detected 
    return False
    ##################################
    raise NotImplementedError


def sample_configuration(map_arr, map_height, map_width, map_resolution, origin_x, origin_y, n_points_to_sample=2000, dim=2):
    ####### your code goes here #######
    x_min = -17
    x_max = 17
    y_min = -7
    y_max = 15
    sampled_points = set() #unordered collection of unique elements: each point added to the set is unique
    while len(sampled_points) < n_points_to_sample:
        #generate rnd coordinates within the map's bounding box 
        x_rand = np.random.uniform(x_min, x_max)
        y_rand = np.random.uniform(y_min, y_max)

        #convert rnd coord to map coord
        # y_map, x_map = pose2map_coordinates(map_resolution, origin_x, origin_y, x_rand, y_rand)
        # print(x_map, y_map)

        #add the sampled point if it belongs to the range (boundaries) + if it's safe 
        if dim == 2: 
            sampled_points.add((x_rand, y_rand))
        elif dim == 3: 
            theta_rand = np.random.uniform(-np.pi,np.pi) 
            sampled_points.add((x_rand, y_rand, theta_rand))
    return sampled_points 

    ##################################
    raise NotImplementedError


def create_prm_traj(map_file):

    def create_prm_graph(number_of_samples):
        # TODO: create PRM graph
        V = []
        E = []
        i = 0

        # 2: V <- sample_free(X,n)
        while i < number_of_samples: # until we get 100 valid samples
            current_sample = list(sample_configuration(map_arr, map_height, map_width, map_resolution, origin_x, origin_y, n_points_to_sample=1, dim=3))[0]

            if not collision_check(map_arr, map_height, map_width, map_resolution, origin_x, origin_y, x=current_sample[0], y=current_sample[1], theta=current_sample[2]):# theta is redundant
                V.append(current_sample)
                i+=1
        # save vertices in kd-tree
        verticies = KDTree(V)

        index1 = 0
        for point1 in V:
            _, index2 = verticies.query(point1, k=5, p=2) 

            for j in range(1,5):
                point2 = verticies.data[index2[j]]

                # collision check of the edege
                if not has_collsion_edge(point1, point2, 10):
                    E.append((index1,int(index2[j]))) 
            
            index1+=1
        
        # TODO: create PRM trajectory (x,y) saving it to prm_traj list

        graph = {
        'nodes': V,
        'edges': E
        }  

        return graph

    def has_collsion_edge(point1, point2, number_of_samples):
        new_point = []

        # sample points along edge
        for i in np.linspace(0,1,number_of_samples):
            new_point.append(point1[0] + (point2[0]-point1[0]) * i)
            new_point.append(point1[1] + (point2[1]-point1[1]) * i)
            new_point.append(point1[2] + (point2[2]-point1[2]) * i)
            if (collision_check(map_arr, map_height, map_width, map_resolution, origin_x, origin_y, x=new_point[0], y=new_point[1], theta=new_point[2])):
                return True
            new_point = []
    
        return False
    
    # Query for a motion plan
    def plan(x_start, x_goal):

        # Connect start and goal        
        connect_point(x_start)
        connect_point(x_goal)

        # Find shortest path
        node1 = (0, 0, 0)
        node2 = (0, 1, 0)
        node3 = (0, 2, 0)
        node4 = (0,0.5,0)
        test_graph = {
        'nodes': np.array([node1, node2, node3, node4]),
        'edges':[(0,1),(1, 2), (2, 3)]}

        solver = A_star(test_graph)
        solver.a_star((0,0,0),(0,2,0))

        # Return result 
        return 0

    def connect_point(point):
        V.append(point)
        index_point = len(V)-1

        # Connect point to graph
        has_found = False
        i=2
        while not has_found:
            distance, index2 = verticies.query(point, k=i, p=2) 

            # collision check of the edege
            if not has_collsion_edge(point, verticies.data[index2[i-1]], 10):
                E.append((int(index2[i-1]), index_point)) 
                has_found = True
            
            i += 1

        return 0
        

    prm_traj = []
    mid_points = np.array([[0,0,0],
                           [9.5,4.5,np.pi/2],
                           [0,8.5,np.pi],
                           [-13.5,4.5,-np.pi/2]])
    map_arr, map_height, map_width, map_resolution, origin_x, origin_y = load_map_and_metadata(map_file)
    ####### your code goes here #######
    # TODO: create PRM graph
    V = []
    E = []
    i = 0

    # 2: V <- sample_free(X,n)
    while i < 100: # until we get 100 valid samples
        current_sample = list(sample_configuration(map_arr, map_height, map_width, map_resolution, origin_x, origin_y, n_points_to_sample=1, dim=3))[0]

        if not collision_check(map_arr, map_height, map_width, map_resolution, origin_x, origin_y, x=current_sample[0], y=current_sample[1], theta=current_sample[2]):# theta is redundant
            V.append(current_sample)
            i+=1
    # save vertices in kd-tree
    verticies = KDTree(V)

    index1 = 0
    for point1 in V:
        distance, index2 = verticies.query(point1, k=5, p=2) 

        for j in range(1,5):
            point2 = verticies.data[index2[j]]

            # collision check of the edege
            if not has_collsion_edge(point1, point2, 10):
                E.append((index1,int(index2[j]))) 
        
        index1+=1
    
    # TODO: create PRM trajectory (x,y) saving it to prm_traj list

    graph = {
    'nodes': V,
    'edges': E,
    # 'costs': {(node1_index, node2_index): cost1, (node2_index, node3_index): cost2, ...}
    }    

    # Plot graph
    for point in V:
        plt.plot(point[0],point[1],marker="o", markersize=5, color="green")

    for edge in E:
        x1 = V[edge[0]][0]
        x2 = V[edge[1]][0]
        y1 = V[edge[0]][1]
        y2 = V[edge[1]][1]
        plt.plot([x1,x2],[y1,y2],'k-')


    plt.plot(mid_points[0][0],mid_points[0][1],marker="o", markersize=5, color="red", label="start")
    plt.plot(mid_points[1][0],mid_points[1][1],marker="o", markersize=5, color="red", label="goal")
    
    plt.show()

    plan(mid_points[1], mid_points[2])
    # prm_traj = np.concatenate(prm_traj, axis=0)
    # np.save(os.path.join(pathlib.Path(__file__).parent.resolve().parent.resolve(),'resource/prm_traj.npy'), prm_traj)

    ##################################
    

def sample_conrol_inputs(number_of_samples=10):
    ####### your code goes here #######
    velocity_samples = np.random.uniform(low=-0.5, high=0.5, size=number_of_samples) # 0.5 as max (based on lab1)
    steering_angle_samples =  np.random.uniform(low=-0.2999998, high=0.2999998, size=number_of_samples) # Generated using left and right commands with the keyboard teleop

    samples = np.array([velocity_samples,steering_angle_samples])

    return samples
    ##################################
    raise NotImplementedError


def forward_simulation_of_kineamtic_model(x, y, theta, v, delta, dt=0.5):
    ####### your code goes here #######
    distance = 0.0381 # distance between the wheel-axises -> this distance is taken from "ego_racecar.xacro"

    # Define the system dynamics nonlinear function
    x_new = x + v * np.cos(theta) * dt
    y_new = y + v * np.sin(theta) * dt
    theta_new = theta + v * np.tan(delta) * 1/distance * dt
    
    ##################################
    # raise NotImplementedError
    return x_new, y_new, theta_new


def create_kino_rrt_traj(map_file):
    kino_rrt_traj = []
    mid_points = np.array([[0,0,0],
                           [9.5,4.5,np.pi/2],
                           [0,8.5,np.pi],
                           [-13.5,4.5,-np.pi/2]])
    map_arr, map_hight, map_width, map_resolution, origin_x, origin_y = load_map_and_metadata(map_file)
    ####### your code goes here #######    
    # TODO: create RRT graph and find the path saving it to kino_rrt_traj list
    


    ##################################
    
    kino_rrt_traj = np.array(kino_rrt_traj)
    np.save(os.path.join(pathlib.Path(__file__).parent.resolve().parent.resolve(),'resource/kino_rrt_traj.npy'), kino_rrt_traj)


if __name__ == "__main__":
    map_file = 'maps/levine.png'
    # create_prm_traj(map_file)
    create_kino_rrt_traj(map_file)
