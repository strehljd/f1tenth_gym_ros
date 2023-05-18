import numpy as np
from scipy.spatial import KDTree
from PIL import Image
import yaml
import os
import pathlib
from a_star import A_star, Node


def load_map_and_metadata(map_file):
    # load the map from the map_file
    map_img = Image.open(map_file).transpose(Image.FLIP_TOP_BOTTOM)
    map_arr = np.array(map_img, dtype=np.uint8)
    map_arr[map_arr < 220] = 1
    map_arr[map_arr >= 220] = 0
    map_arr = map_arr.astype(np.bool)
    map_hight = map_arr.shape[0]
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
        
    return map_arr, map_hight, map_width, map_resolution, origin_x, origin_y


def pose2map_coordinates(map_resolution, origin_x, origin_y, x, y):
    x_map = int((x - origin_x) / map_resolution)
    y_map = int((y - origin_y) / map_resolution)
    return y_map, x_map


def collision_check(map_arr, map_hight, map_width, map_resolution, origin_x, origin_y, x, y, theta):
    ####### your code goes here #######
    # TODO: transform configuration to workspace bounding box
    
    # TODO: overlay workspace bounding box on map (creating borders for collision search in the next step)
    
    # TODO: check for collisions by looking inside the bounding box on the map if there are values greater than 0
    
    ##################################
    raise NotImplementedError


def sample_configuration(map_arr, map_hight, map_width, map_resolution, origin_x, origin_y, n_points_to_sample=2000):
    ####### your code goes here #######
    
    
    
    ##################################
    raise NotImplementedError


def create_prm_traj(map_file):
    prm_traj = []
    
    ####### your code goes here #######
    # TODO: load the map and metadata
    
    # TODO: create PRM graph
    
    # TODO: create PRM trajectory (x,y) saving it to prm_traj list
    
    ##################################
    
    prm_traj = np.array(prm_traj)
    np.save(os.path.join(pathlib.Path(__file__).parent.resolve().parent.resolve(),'resource/prm_traj.npy'), prm_traj)


def forward_simulation_of_kineamtic_model(x, y, theta, v, delta, dt):
    ####### your code goes here #######
    
    
    
    ##################################
    raise NotImplementedError
    return x_new, y_new, theta_new


def create_kino_rrt_traj(map_file):
    kino_rrt_traj = []
    
    ####### your code goes here #######
    # TODO: load the map and metadata
    
    # TODO: create RRT graph and find the path saving it to kino_rrt_traj list
    
    ##################################
    
    kino_rrt_traj = np.array(kino_rrt_traj)
    np.save(os.path.join(pathlib.Path(__file__).parent.resolve().parent.resolve(),'resource/kino_rrt_traj.npy'), kino_rrt_traj)


if __name__ == "__main__":
    map_file = 'maps/levine.png'
    data = load_map_and_metadata(map_file)
    print(data)
    # create_prm_traj(map_file)
    # create_kino_rrt_traj(map_file)
