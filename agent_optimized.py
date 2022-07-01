import time
import kuimaze
import os
import random
import queue

class Agent(kuimaze.SearchAgent):
    def __init__(self, environment):
        self.environment = environment
 
    def heuristic_function(self, curr, last):
        reward = 0
        vector = [curr[0] - last[0], curr[1] - last[1]]
        z_axis = vector[0] * self.environment._grad[0] + vector[1] * self.environment._grad[1]
        if curr != last:
            reward = abs(vector[0]) + abs(vector[1]) + z_axis
        return reward
 
    def find_path(self):
        settings = self.environment.reset() #maze initialization
        start = settings[0][0:2] #starting position
        goal = settings[1][0:2] #goals position
 
        memory = {start : [0, (-1, -1)] } #memory: position, dist from start, previous node
        alreadyInPQ = [start] #already in PQ
        PQ = queue.PriorityQueue()
        PQ.put((0, start)) #PQ = (fscore, position), where fscore is distance from start + heuristic approximation of the cost to reach goal state
 
        while not PQ.empty():
            current_position = PQ.get()[1] 
 
            if current_position == goal: #break the loop when the goal position is reached
                path = [goal] #reconstruct path 
                while current_position != start:
                    current_position = memory[current_position][1]
                    path.append(current_position)
                path.reverse() #path is in reversed order at this moment
                return path #return found path
 
            dist_from_start = memory[current_position][0]
            for pos, next_node_dist in self.environment.expand(current_position):
                if pos not in alreadyInPQ: #if this position wasnt already inserted into PQ, insert it
                    fscore = dist_from_start + next_node_dist + self.heuristic_function(pos, goal) #because costs are defined in kui function and manhattan/euclid doesnt work
                    memory[pos] = [dist_from_start + next_node_dist, current_position] #remember the previous node for path reconstruction
                    PQ.put((fscore, pos)) #insert into PQ
                    alreadyInPQ.append(pos) #prevent from reinserting into PQ
        return None

if __name__ == '__main__':
 
    MAP = 'maps/easy/easy1.bmp'
    MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), MAP)
    GRAD = (0, 0)
    SAVE_PATH = False
    SAVE_EPS = False
 
 
 
    env = kuimaze.InfEasyMaze(map_image=MAP, grad=GRAD)       # For using random map set: map_image=None
    agent = Agent(env) 
 
    path = agent.find_path()
    env.set_path(path)          # set path it should go from the init state to the goal state
    if SAVE_PATH:
        env.save_path()         # save path of agent to current_position directory
    if SAVE_EPS:
        env.save_eps()          # save rendered image to eps
    env.render(mode='human')
    time.sleep(3)