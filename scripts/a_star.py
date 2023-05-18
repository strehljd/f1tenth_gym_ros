import numpy as np
from typing import Callable
from heapdict import heapdict


class Node():
    def __init__(self, pose, h, parent=None, cost=0):
        self.pose = pose
        self.parent = parent
        
    def __eq__(self, other):
        return self.pose == other.pose
    
    def __hash__(self):
        return hash(self.pose)


######### Graph structure #########
# graph = {
#     'nodes': [node1, node2, ...],
#     'edges': [(node1, node2), (node2, node3), ...],
#     'costs': {(node1, node2): cost1, (node2, node3): cost2, ...}
# node = (x, y, theta)


class A_star():
    def __init__(self, graph: dict, successor_fn: Callable = None):
        self.graph = graph
        # check if graph is connected (if has key called edges and that key's value is not an empty list)
        if not self.graph.get('edges') or not self.graph['edges']:
            self.successor_fn = successor_fn
        else:
            self.successor_fn = self.successor_fn_graph
            
        if not self.graph.get('costs') or not self.graph['costs']:
            self.graph['costs'] = {}
            for edge in self.graph['edges']:
                self.graph['costs'][edge] = np.linalg.norm(np.array(edge[0][:2]) - np.array(edge[1][:2]))
            
    def successor_fn_graph(self, node):
        node_pose = node.pose
        edges = np.array(self.graph['edges'])
        successor_poses = edges[edges[:,0] == node_pose][:,1]
        node_edges = edges[edges[:,0] == node_pose]
        successors = [Node(pose, node) for pose in successor_poses]
        successor_costs = [self.graph['costs'][tuple(edge)] for edge in node_edges]
        return successors, successor_costs
    
    def heuristic(self, node, goal):
        return np.linalg.norm(np.array(node[:2]) - np.array(goal[:2]))
    
    def check_goal_condition(self, node, goal):
        if np.linalg.norm(np.array(node[:2]) - np.array(goal[:2])) < 0.5:
            return True
    
    def a_star(self, start, goal):
        # initialize the open and closed lists
        open_list = heapdict()
        closed_list = {}
        # add the start node
        start_node = self.graph['nodes'][np.argmin(np.linalg.norm(np.array(self.graph['nodes'])[:2] - np.array(start[:2]), axis=1))]
        open_list[start_node] = self.heuristic(start_node.pose, goal)
        # while the open list is not empty
        while open_list:
            # get the current node
            current_node, node_f_val = open_list.popitem()
            node_g = node_f_val - self.heuristic(current_node.pose, goal)
            # add the current node to the closed list
            closed_list[current_node] = node_g  # g
            # if we have reached the goal, return the path
            if self.check_goal_condition(current_node.pose, goal):
                path = []
                while current_node != start_node:
                    path.append(current_node.pose[:2])
                    current_node = current_node.parent
                path.append(start[:2])
                path.reverse()
                return path
            # expand the current node
            successors, successor_costs = self.successor_fn(current_node)
            for successor, successor_cost in zip(successors, successor_costs):
                new_g = node_g + successor_cost
                # if the successor is new
                if successor not in closed_list.keys() or successor not in open_list.keys():
                    # add it to the open list
                    open_list[successor] = new_g + self.heuristic(successor.pose, goal)
                # otherwise, check if we have a better path
                elif successor in open_list.keys():
                    curr_successor, curr_g = [(node, f - self.heuristic(node.pose, goal)) for node, f in open_list.items() if node == successor][0]
                    # if this path is better, update the parent and cost
                    if new_g < curr_g:
                        open_list.pop(successor, None)
                        curr_successor.parent = current_node
                        open_list[curr_successor] = new_g + self.heuristic(successor.pose, goal)
                else: # successor in closed_list.keys()
                    curr_successor, curr_g = [(node, g) for node, g in closed_list.items() if node == successor][0]
                    # if this path is better, update the parent and cost
                    if new_g < curr_g:
                        closed_list.pop(successor, None)
                        curr_successor.parent = current_node
                        open_list[curr_successor] = new_g + self.heuristic(successor.pose, goal)
                        
        return None
        
            
        
        