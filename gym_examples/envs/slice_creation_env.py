import gymnasium as gym
#from gym import spaces
import pygame
import numpy as np
import pandas as pd
from stable_baselines3.common.env_checker import check_env
from copy import deepcopy


class SliceCreationEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        # Define environment parameters
        
        #MEC 1, 2, 3 available resources
        self.mec_bw = [1000, 1000, 1000]
        
        self.slices_param = [10, 20, 50]

        self.slice_requests = pd.read_csv('/home/mario/Documents/DQN_Models/Model 1/gym-examples/gym_examples/slice_request_db1')  # Load VNF requests from the generated CSV
        
        self.observation_space = gym.spaces.Box(low=0, high=1000, shape=(5,), dtype=np.float32)
        
        self.action_space = gym.spaces.Discrete(4)  # 0: Do Nothing, 1: Allocate MEC 1, 2: Allocate MEC 2, 3: Allocate MEC 3

        #self.process_requests()
        
        # Define other necessary variables and data structures
        self.current_time_step = 1
        self.reward = 0
        self.first = True
        
        self.processed_requests = []

    def reset(self, seed=None, options=None):
        # Initialize the environment to its initial state
        
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        self.current_time_step = 1
        self.reward = 0
        self.processed_requests = []
        next_request = self.read_request()
        self.update_slice_requests(next_request)
        self.observation = np.array([next_request[1], next_request[2]] + deepcopy(self.mec_bw), dtype=np.float32)
        #self.observation = np.array(self.observation, np.float32)
        self.info = {}
        self.first = True
        
        return self.observation, self.info



    def step(self, action):
        
        if self.first:
            next_request = self.processed_requests[0]
            self.first = False
        else: 
            next_request = self.read_request()
            self.update_slice_requests(next_request)
        terminated = False
        
        # Apply the selected action (0: Do Nothing, 1: Allocate MEC 1, 2: Allocate MEC 2, 3: Allocate MEC 3)
        
        if action == 1 and next_request[1] == 1:
            if self.check_resources(next_request[2], next_request[1]):
                self.allocate_slice(next_request[2], next_request[1])
                self.create_slice(next_request)
                self.reward += 1       
            else: terminated = True
        
        if action == 1 and next_request[1] != 1:
            terminated = True
            
        if action == 2 and next_request[1] == 2:
            if self.check_resources(next_request[2], next_request[1]):
                self.allocate_slice(next_request[2], next_request[1])
                self.create_slice(next_request)
                self.reward += 1       
            else: terminated = True
        
        if action == 2 and next_request[1] != 2:
            terminated = True
            
        if action == 3 and next_request[1] == 3:
            if self.check_resources(next_request[2], next_request[1]):
                self.allocate_slice(next_request[2], next_request[1])
                self.create_slice(next_request)
                self.reward += 1       
            else: terminated = True
        
        if action == 3 and next_request[1] != 3:
            terminated = True
            
        if action == 0:
            if not self.check_resources(next_request[2], next_request[1]):
                self.reward += 1
            else: terminated = True        
    
        observation = np.array([next_request[1], next_request[2]] + self.mec_bw, dtype=np.float32)
        
        reward = self.reward
        
        #done = False
        
        info = {}  # Additional information (if needed)
        
        #self.current_time_step += 1  # Increment the time step
        
        return observation, reward, terminated, False, info
    
    def read_request(self):
        next_request = self.slice_requests.iloc[self.current_time_step - 1]
        request_list =list([next_request['ARRIVAL_REQUEST_@TIME'], next_request['MEC_ID'], next_request['SLICE_BW_REQUEST'], next_request['SLICE_KILL_@TIME']])
        self.current_time_step += 1
        return request_list
        

    def update_slice_requests(self, request):
        # Update the slice request list by deleting the killed VNFs
        if len(self.processed_requests) != 0:
            for i in self.processed_requests:
                #i[3] < request[0]
                if i[3] < request[0]:
                    self.deallocate_slice(i)
        self.processed_requests.append(request)
        

    def check_resources(self, slice_bw_request, mec_id):
        # Logic to check if there are available resources to allocate the VNF request
        # Return True if resources are available, False otherwise
        if self.mec_bw[int(mec_id)-1] >= int(slice_bw_request):
            return True
        else: return False
    
    def allocate_slice(self, slice_bw_request, mec_id):
        # Allocate the resources requested by the current VNF
        self.mec_bw[int(mec_id)-1] -= int(slice_bw_request)
        # Define Slice ID
    
    def deallocate_slice(self, request):
        # Function to deallocate resources of killed requests
        self.mec_bw[int(request[1]-1)] = self.mec_bw[int(request[1]-1)] + request[2]
        
    def create_slice (self, request):
        # Function to create the slice for a specific request
        # This function inserts the defined slice to the request in the processed requests list
        
        resources = request[2]
        if resources <= self.slices_param[0]:
            slice_id = 1
        elif resources <= self.slices_param[1]:
            slice_id = 2
        elif resources <= self.slices_param[2]:
            slice_id = 3
        
        self.processed_requests[len(self.processed_requests) - 1].append(slice_id)
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            
#a = SliceCreationEnv()
#check_env(a)