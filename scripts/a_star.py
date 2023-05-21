import numpy as np
from typing import Callable
from heapdict import heapdict


class Node():
    def __init__(self, pose, idx, parent_id=None, cost=0):
        self.pose = pose
        self.parent_id = parent_id
        self.idx = idx
        
    def __eq__(self, other):
        return self.pose == other.pose
    
    def __hash__(self):
        return hash(self.idx)


######### Graph structure #########
# graph = {
#     'nodes': [node1, node2, ...],
#     'edges': [(node1_index, node2_index), (node2_index, node3_index), ...],
#     'costs': {(node1_index, node2_index): cost1, (node2_index, node3_index): cost2, ...}
# node = (x, y, theta)


class A_star():
    def __init__(self, graph: dict, successor_fn: Callable = None):
        self.graph = graph
        # check if graph is connected (if has key called edges and that key's value is not an empty list)
        if not self.graph.get('edges') or not self.graph['edges']:
            if successor_fn:
                self.successor_fn = successor_fn
            else:
                raise ValueError('Graph is not connected and no successor function was provided.')
        else:
            self.successor_fn = self.successor_fn_graph
            
        if not self.graph.get('costs') or not self.graph['costs']:
            self.graph['costs'] = {}
            for edge in self.graph['edges']:
                self.graph['costs'][edge] = np.linalg.norm(np.array(self.graph['nodes'][edge[0]][:2]) - np.array(self.graph['nodes'][edge[1]][:2]))
            
    def successor_fn_graph(self, node):
        node_pose = node.pose
        edges = np.array(self.graph['edges'])
        successor_ids = edges[np.linalg.norm(np.array(self.graph['nodes'][edges[:,0]]) - node_pose, axis=1) == 0][:,1]
        successor_poses = self.graph['nodes'][successor_ids]
        node_edges = edges[np.linalg.norm(np.array(self.graph['nodes'][edges[:,0]]) - node_pose, axis=1) == 0]
        successors = [Node(pose, id, node.idx) for pose, id in zip(successor_poses, successor_ids)]
        successor_costs = [self.graph['costs'][tuple(edge)] for edge in node_edges]
        return successors, successor_costs
    
    def heuristic(self, node, goal):
        return np.linalg.norm(np.array(node[:2]) - np.array(goal[:2]))
    
    def check_goal_condition(self, node, goal):
        if np.linalg.norm(np.array(node[:2]) - np.array(goal[:2])) < 1:
            return True
    
    def a_star(self, start, goal):
        # initialize the open and closed lists
        open_list = heapdict()
        closed_list = {}
        # add the start node
        start_idx = np.argmin(np.linalg.norm(np.array(self.graph['nodes']) - np.array(start), axis=1))
        start_pose = self.graph['nodes'][np.argmin(np.linalg.norm(np.array(self.graph['nodes']) - np.array(start), axis=1))]
        start_node = Node(start_pose, start_idx)
        open_list[start_idx] = (self.heuristic(start_node.pose, goal), start_node)
        # while the open list is not empty
        while open_list:
            # get the current node
            node_f_val, current_node = open_list.popitem()[1]
            node_g = node_f_val - self.heuristic(current_node.pose, goal)
            # add the current node to the closed list
            closed_list[current_node.idx] = (current_node, node_g)
            # if we have reached the goal, return the path
            if self.check_goal_condition(current_node.pose, goal):
                path = []
                while current_node.idx != start_node.idx:
                    path.append(current_node.pose)
                    current_node = closed_list[current_node.parent_id][0]
                # path.append(start)
                path.reverse()
                return path
            # expand the current node
            successors, successor_costs = self.successor_fn(current_node)
            for successor, successor_cost in zip(successors, successor_costs):
                new_g = node_g + successor_cost
                # if the successor is new
                closed_list_ids = np.array(list(closed_list.keys()))
                open_list_ids = np.array(list(open_list.keys()))
                if successor.idx not in closed_list_ids and successor.idx not in open_list_ids:
                    # add it to the open list
                    open_list[successor.idx] = (new_g + self.heuristic(successor.pose, goal), successor)
                    continue
                # otherwise, check if we have a better path
                elif successor.idx in open_list_ids:
                    curr_successor, curr_g = [(node, f - self.heuristic(node.pose, goal)) for f, node in list(open_list.values()) if node.idx == successor.idx][0]
                    # if this path is better, update the parent and cost
                    if new_g < curr_g:
                        open_list.pop(curr_successor.idx, None)
                        open_list[successor.idx] = (new_g + self.heuristic(successor.pose, goal), successor)
                        
        return []
        
            
        
        