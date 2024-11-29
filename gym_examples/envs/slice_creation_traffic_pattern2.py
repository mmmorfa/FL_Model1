import gymnasium as gym
#from gym import spaces
import pygame
import numpy as np
import pandas as pd
from stable_baselines3.common.env_checker import check_env
from copy import deepcopy
from random import randint
from math import log2, ceil, floor
import sqlite3
import json
import datetime



# Pandas config
pd.set_option("display.max_rows", None, "display.max_columns", None)

# ****************************** VNF Generator GLOBALS ******************************
# File directory
db_name = '/home/mario/Documents/FL_Model1-main/gym_examples/slice_request_db{}'.format(datetime.datetime.now())
DIRECTORY = db_name

# Number of VNF types dictionary
# i.e. {key: value}, value = [MEC_CPU, MEC_RAM, MEC_Storage, MEC_BW, RAN_R, RAN_L]
VNF_TYPES = {0: [2, 4, 10, 8, 15, 7], 1: [4, 8, 32, 20, 45, 1], 2: [4, 8, 20, 14, 30, 1], 3: [2, 4, 8, 5, 10, 8], 
             4: [4, 12, 64, 30, 50, 15], 5: [1, 2, 5, 2, 8, 35], 6: [2, 4, 10, 10, 5, 40], 7: [4, 16, 64, 30, 54, 20], 
             8: [2, 16, 20, 25, 35, 25], 9: [4, 8, 25, 15, 40, 30], 10: [4, 4, 16, 20, 25, 2], 11: [4, 8, 16, 25, 28, 5]}
# Suburban Moderate Mobility
ARRIVAL_RATE = {0: 3, 1: 2, 2: 3, 3: 3, 4: 2, 5: 3, 6: 3, 7: 2, 8: 3, 9: 3, 10: 2, 11: 3}
LIFE_CYCLE_RATE = {0: 8, 1: 6, 2: 5, 3: 4, 4: 6, 5: 8, 6: 8, 7: 6, 8: 5, 9: 4, 10: 6, 11: 8}
# Num of vnf requests
NUM_VNF_REQUESTS = 100

# ****************************** VNF Generator FUNCTIONS ******************************

def generate_requests_per_type(key, num):
    """ This function generates a set of requests per type """
    req = []
    vnf_request_at_time = 0

    x = 0  # to check the inter arrival times
    y = 0  # to check the holding times

    for _ in range(num):
        # Generate inter-arrival time for the VNF request
        inter_vnf_time_request = np.random.exponential(1.0 / ARRIVAL_RATE[key])
        # Update the time for the next request
        vnf_request_at_time += inter_vnf_time_request

        # Generate holding time for the VNF request
        #vnf_request_life_time = np.random.exponential(LIFE_CYCLE_RATE[key])
        # Alternative: Use a Poisson distribution for holding time
        vnf_request_life_time = np.random.poisson(LIFE_CYCLE_RATE[key]) 
        vnf_kill_at_time = vnf_request_at_time + vnf_request_life_time

        final_vnf = [vnf_request_at_time, VNF_TYPES[key][0],VNF_TYPES[key][1],VNF_TYPES[key][2],VNF_TYPES[key][3],VNF_TYPES[key][4],VNF_TYPES[key][5], vnf_kill_at_time]
        #final_vnf = [vnf_request_at_time, VNF_TYPES[key][0], vnf_kill_at_time]

        # Round up decimals
        final_vnf = [round(val, 3) if isinstance(val, (int, float)) else val for val in final_vnf]
        req.append(final_vnf)

        x += inter_vnf_time_request
        y += vnf_request_life_time

    # print("DEBUG: key = ", key, "average inter-arrival = ", x / num, "average holding = ", y / num)
    return req


def get_key(val):
    """ Get value key """
    for k, v in VNF_TYPES.items():
        if val == v:
            return k

def generate_vnf_list():
    # ****************************** MAIN CODE ******************************
    # The overall procedure to create the requests is as follows:
        # - generate a set of requests per type
        # - put them altogether
        # - sort them according the arrival time
        # - return the num_VNFs_requests number of them '''

    vnfList = []

    for vnf in list(VNF_TYPES.values()):
        # Get vnf key for the arrival and holding dicts
        key = get_key(vnf)

            # We don't know how many requests from each type will be in the final list of requests.
            # It depends on the arrival rate of the type in comparison to the rates of the other types.
            # So, we generate the maximum number, i.e., num_VNFs_requests.
        requests = generate_requests_per_type(key, NUM_VNF_REQUESTS)

            # vnfList will be all the requests from all types not sorted.
        for req in requests:
            vnfList.append(req)

    # Sort the requests according to the arrival rate
    vnfList.sort(key=lambda x: x[0])

        # Until now, we have generated num_VNFs_requests * len(vnf_types) requests.
        # We only need the num_VNFs_requests of them'''
    vnfList = vnfList[:NUM_VNF_REQUESTS]

        # Dataframe
    columns = ['ARRIVAL_REQUEST_@TIME','SLICE_MEC_CPU_REQUEST', 'SLICE_MEC_RAM_REQUEST', 'SLICE_MEC_STORAGE_REQUEST', 'SLICE_MEC_BW_REQUEST', 'SLICE_RAN_R_REQUEST','SLICE_RAN_L_REQUEST', 'SLICE_KILL_@TIME']
    df = pd.DataFrame(data=vnfList, columns=columns, dtype=float)

        # Export df to  csv file
    df.to_csv(DIRECTORY, index=False, header=True)

#------------------------------------Environment Class----------------------------------------------------
class SliceCreationEnv5(gym.Env):
    metadata = {"render_modes": [], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        # Define environment parameters

        generate_vnf_list()

        # RAN Global Parameters -------------------------BWP  1------------------------------------------------------------------------------------------------------------------------------------
        self.numerology1 = 0                       # 0,1,2,3,...
        self.scs1 = 2**(self.numerology1) * 15_000   # Hz
        self.slot_per_subframe1 = 2**(self.numerology1)
        
        self.channel_BW1 = 50_000_000              # Hz (100MHz for <6GHz band, and 400MHZ for mmWave)
        self.guard_BW1 = 692_500                   # Hz (for symmetric guard band)

        self.PRB_BW1 = self.scs1 * 12               # Hz - Bandwidth for one PRB (one OFDM symbol, 12 subcarriers)
        self.PRB_per_channel1 = floor((self.channel_BW1 - (self.guard_BW1)) / (self.PRB_BW1))        # Number of PRB to allocate within the channel bandwidth
        self.spectral_efficiency1 = 5.1152        # bps/Hz (For 64-QAM and 873/1024)

        self.PRB_map1 = np.zeros((14, self.PRB_per_channel1))                                 # PRB map per slot (14 OFDM symbols x Number of PRB to allocate within the channel bandwidth)

        # RAN Global Parameters -------------------------BWP  2------------------------------------------------------------------------------------------------------------------------------------
        self.numerology2 = 1                       # 0,1,2,3,...
        self.scs2 = 2**(self.numerology2) * 15_000   # Hz
        self.slot_per_subframe2 = 2**(self.numerology2)
        
        self.channel_BW2 = 50_000_000              # Hz (100MHz for <6GHz band, and 400MHZ for mmWave)
        self.guard_BW2 = 1_045_000                   # Hz (for symmetric guard band)

        self.PRB_BW2 = self.scs2 * 12               # Hz - Bandwidth for one PRB (one OFDM symbol, 12 subcarriers)
        self.PRB_per_channel2 = floor((self.channel_BW2 - (self.guard_BW2)) / (self.PRB_BW2))        # Number of PRB to allocate within the channel bandwidth
        self.spectral_efficiency2 = 5.1152        # bps/Hz (For 64-QAM and 873/1024)

        self.PRB_map2 = np.zeros((14, self.PRB_per_channel2))                                 # PRB map per slot (14 OFDM symbols x Number of PRB to allocate within the channel bandwidth)



        #Available MEC resources (Order: MEC_CPU (Cores), MEC_RAM (GB), MEC_STORAGE (GB), MEC_BW (Mbps))------------------------------------------------------------------------------------
        #self.resources = [1000]
        self.resources_1 = {'MEC_CPU': 30, 'MEC_RAM': 128, 'MEC_STORAGE': 1000, 'MEC_BW': 300}
        self.resources_2 = {'MEC_CPU': 30, 'MEC_RAM': 128, 'MEC_STORAGE': 1000, 'MEC_BW': 200}
        self.resources_3 = {'MEC_CPU': 50, 'MEC_RAM': 128, 'MEC_STORAGE': 1000, 'MEC_BW': 200}
        self.resources_4 = {'MEC_CPU': 30, 'MEC_RAM': 128, 'MEC_STORAGE': 1000, 'MEC_BW': 200}
        self.resources_5 = {'MEC_CPU': 20, 'MEC_RAM': 64, 'MEC_STORAGE': 800, 'MEC_BW': 100}
        self.resources_6 = {'MEC_CPU': 20, 'MEC_RAM': 64, 'MEC_STORAGE': 800, 'MEC_BW': 80}
        
        #Defined parameters per Slice. (Each component is a list of the correspondent slice parameters)-------------------------------------------------------------------------------------
        self.slices_param = {1: [4, 16, 100, 40, 50, 20], 2: [4, 32, 100, 100, 30, 30], 3: [8, 16, 32, 80, 20, 1], 
                             4: [4, 8, 16, 50, 25, 5], 5: [2, 8, 32, 40, 10, 10], 6: [2, 8, 32, 40, 5, 40]}

        self.slice_requests = pd.read_csv(DIRECTORY)  # Load VNF requests from the generated CSV
        #self.slice_requests = pd.read_csv('/data/scripts/DQN_models/Model1/gym_examples/slice_request_db1')    #For pod
        
        # VECTORS----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.observation_space = gym.spaces.Box(low=0, high=10000, shape=(7,), dtype=np.float32) #observation space composed by Requested resources (MEC_CPU, MEC_RAM, MEC_ST, MEC_BW, RAN_R, RAN_L) and the available resource flag.
        
        self.action_space = gym.spaces.Discrete(7)  # 0: Do Nothing, 1: Allocate Slice 1, 2: Allocate Slice 2, 3: Allocate Slice 3, 4: Allocate Slice 4, 5: Allocate Slice 5, 6: Allocate Slice 6

        #self.process_requests()
        
        # Other necessary variables and data structures
        self.current_time_step = 1
        self.reward = 0
        self.first = True
        self.resources_flag = 1
        
        self.processed_requests = []

        # Create Database 
        #self.create_db()

    #-------------------------Reset Method----------------------------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        # Initialize the environment to its initial state

        #print(self.processed_requests)

        generate_vnf_list()

        # Create Database 
        #self.create_db()

        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        self.current_time_step = 1
        self.reward = 0

        self.processed_requests = []
        #self.update_db('processed_requests', 0)

        self.reset_resources()

        self.slice_requests = pd.read_csv(DIRECTORY)  # Load VNF requests from the generated CSV
        #self.slice_requests = pd.read_csv('/data/scripts/DQN_models/Model1/gym_examples/slice_request_db1')    #For pod

        self.next_request = self.read_request()
        #slice_id = self.create_slice(self.next_request)
        self.update_slice_requests(self.next_request)
        self.check_resources(self.next_request)
        self.observation = np.array([self.next_request['SLICE_MEC_CPU_REQUEST']] + [self.next_request['SLICE_MEC_RAM_REQUEST']] 
                                    + [self.next_request['SLICE_MEC_STORAGE_REQUEST']] + [self.next_request['SLICE_MEC_BW_REQUEST']] 
                                    + [self.next_request['SLICE_RAN_R_REQUEST']]+ [self.next_request['SLICE_RAN_L_REQUEST']] +[self.resources_flag], dtype=np.float32)
        #self.observation = np.array([self.next_request[1]] + deepcopy(self.resources), dtype=np.float32)
        self.info = {}
        self.first = True
        
        #print("\nReset: ", self.observation)
        
        return self.observation, self.info

    #-------------------------Step Method-------------------------------------------------------------------------
    def step(self, action):
        
        #self.read_parameter_db('processed_requests', 0)
        #self.read_parameter_db('PRB_map1', 0)
        #self.read_parameter_db('PRB_map2', 0)

        if self.first:
            #self.read_parameter_db('processed_requests', 0)
            self.next_request = self.processed_requests[0]
            self.first = False
        #else: 
            #next_request = self.read_request()
        #    self.update_slice_requests(self.next_request)
            
        terminated = False
        
        slice_id = self.create_slice(self.next_request)
        
        reward_value = 1
        
        # Apply the selected action (0: Do Nothing, 1: Allocate Slice 1, 2: Allocate Slice 2, 3: Allocate Slice 3)
        terminated = self.evaluate_action(action, slice_id, reward_value, terminated) 

        self.update_slice_requests(self.next_request)

        self.check_resources(self.next_request)
    
        #self.observation = np.array([self.next_request[1]] + [self.resources_flag], dtype=np.float32)
        self.observation = np.array([self.next_request['SLICE_MEC_CPU_REQUEST']] + [self.next_request['SLICE_MEC_RAM_REQUEST']] 
                                    + [self.next_request['SLICE_MEC_STORAGE_REQUEST']] + [self.next_request['SLICE_MEC_BW_REQUEST']] 
                                    + [self.next_request['SLICE_RAN_R_REQUEST']] + [self.next_request['SLICE_RAN_L_REQUEST']] + [self.resources_flag], dtype=np.float32)
        
        #done = False
        
        info = {}  # Additional information (if needed)
        
        #self.current_time_step += 1  # Increment the time step
        
        #print("Action: ", action, "\nObservation: ", self.observation, "\nReward: ", self.reward)

        truncated = False
        
        return self.observation, self.reward, terminated, truncated, info
    
    def read_request(self):
        next_request = self.slice_requests.iloc[self.current_time_step - 1]

        SiNR = randint(1, 20)

        #request_list =list([next_request['ARRIVAL_REQUEST_@TIME'], next_request['SLICE_BW_REQUEST'], next_request['SLICE_KILL_@TIME']])
        request_list = {'ARRIVAL_REQUEST_@TIME': next_request['ARRIVAL_REQUEST_@TIME'], 'SLICE_MEC_CPU_REQUEST': next_request['SLICE_MEC_CPU_REQUEST'], 
                        'SLICE_MEC_RAM_REQUEST': next_request['SLICE_MEC_RAM_REQUEST'], 'SLICE_MEC_STORAGE_REQUEST': next_request['SLICE_MEC_STORAGE_REQUEST'],
                        'SLICE_MEC_BW_REQUEST': next_request['SLICE_MEC_BW_REQUEST'], 'SLICE_RAN_R_REQUEST': next_request['SLICE_RAN_R_REQUEST'], 
                        'SLICE_RAN_L_REQUEST': next_request['SLICE_RAN_L_REQUEST'], 'SLICE_KILL_@TIME': next_request['SLICE_KILL_@TIME'], 'UE_ID': self.current_time_step, 'UE_SiNR': SiNR}
        self.current_time_step += 1
        return request_list
        
    def update_slice_requests(self, request):
        # Update the slice request list by deleting the killed VNFs
        if len(self.processed_requests) != 0:
            for i in self.processed_requests:
                #i[2] < request[0]
                if len(i) == 11 and i['SLICE_KILL_@TIME'] <= request['ARRIVAL_REQUEST_@TIME']:
                    slice_id = self.create_slice(i)
                    self.deallocate_slice(i, slice_id)
                    self.processed_requests.remove(i)
        self.processed_requests.append(request)
        #self.update_db('processed_requests', 0)
        
    def check_resources(self, request):
        # Logic to check if there are available resources to allocate the VNF request
        # Set Resources flag to 1 if resources are available, 0 otherwise

        #Check RAN Resources ----------------------------------------------------------------------------------------------------------------------------
        ran_resources = self.check_RAN(request)


        # Check MEC resources----------------------------------------------------------------------------------------------------------------------------
        slice_id = self.create_slice(request)

        if slice_id == 1:
            #self.read_parameter_db('resources', 1)
            if (self.resources_1['MEC_CPU'] >= request['SLICE_MEC_CPU_REQUEST'] and self.resources_1['MEC_RAM'] >= request['SLICE_MEC_RAM_REQUEST'] and 
                self.resources_1['MEC_STORAGE'] >= request['SLICE_MEC_STORAGE_REQUEST'] and self.resources_1['MEC_BW'] >= request['SLICE_MEC_BW_REQUEST'] and ran_resources
                ):
                self.resources_flag = 1
            else: self.resources_flag = 0
        elif slice_id == 2:
            #self.read_parameter_db('resources', 2)
            if (self.resources_2['MEC_CPU'] >= request['SLICE_MEC_CPU_REQUEST'] and self.resources_2['MEC_RAM'] >= request['SLICE_MEC_RAM_REQUEST'] and 
                self.resources_2['MEC_STORAGE'] >= request['SLICE_MEC_STORAGE_REQUEST'] and self.resources_2['MEC_BW'] >= request['SLICE_MEC_BW_REQUEST'] and ran_resources
                ):
                self.resources_flag = 1
            else: self.resources_flag = 0
        elif slice_id == 3:
            #self.read_parameter_db('resources', 3)
            if (self.resources_3['MEC_CPU'] >= request['SLICE_MEC_CPU_REQUEST'] and self.resources_3['MEC_RAM'] >= request['SLICE_MEC_RAM_REQUEST'] and 
                self.resources_3['MEC_STORAGE'] >= request['SLICE_MEC_STORAGE_REQUEST'] and self.resources_3['MEC_BW'] >= request['SLICE_MEC_BW_REQUEST'] and ran_resources
                ):
                self.resources_flag = 1
            else: self.resources_flag = 0
        elif slice_id == 4:
            #self.read_parameter_db('resources', 4)
            if (self.resources_4['MEC_CPU'] >= request['SLICE_MEC_CPU_REQUEST'] and self.resources_4['MEC_RAM'] >= request['SLICE_MEC_RAM_REQUEST'] and 
                self.resources_4['MEC_STORAGE'] >= request['SLICE_MEC_STORAGE_REQUEST'] and self.resources_4['MEC_BW'] >= request['SLICE_MEC_BW_REQUEST'] and ran_resources
                ):
                self.resources_flag = 1
            else: self.resources_flag = 0
        elif slice_id == 5:
            #self.read_parameter_db('resources', 5)
            if (self.resources_5['MEC_CPU'] >= request['SLICE_MEC_CPU_REQUEST'] and self.resources_5['MEC_RAM'] >= request['SLICE_MEC_RAM_REQUEST'] and 
                self.resources_5['MEC_STORAGE'] >= request['SLICE_MEC_STORAGE_REQUEST'] and self.resources_5['MEC_BW'] >= request['SLICE_MEC_BW_REQUEST'] and ran_resources
                ):
                self.resources_flag = 1
            else: self.resources_flag = 0
        elif slice_id == 6:
            #self.read_parameter_db('resources', 6)
            if (self.resources_6['MEC_CPU'] >= request['SLICE_MEC_CPU_REQUEST'] and self.resources_6['MEC_RAM'] >= request['SLICE_MEC_RAM_REQUEST'] and 
                self.resources_6['MEC_STORAGE'] >= request['SLICE_MEC_STORAGE_REQUEST'] and self.resources_6['MEC_BW'] >= request['SLICE_MEC_BW_REQUEST'] and ran_resources
                ):
                self.resources_flag = 1
            else: self.resources_flag = 0
    
    def allocate_slice(self, request, slice_id):
        # Allocate the resources requested by the current VNF

        # RAN resources part---------------------------------------------------------------
        self.allocate_ran(request)
        
        # MEC resources part---------------------------------------------------------------
        if slice_id == 1:
            #self.read_parameter_db('resources', 1)
            self.resources_1['MEC_CPU'] -= request['SLICE_MEC_CPU_REQUEST']
            self.resources_1['MEC_RAM'] -= request['SLICE_MEC_RAM_REQUEST']
            self.resources_1['MEC_STORAGE'] -= request['SLICE_MEC_STORAGE_REQUEST']
            self.resources_1['MEC_BW'] -= request['SLICE_MEC_BW_REQUEST']
            #self.update_db('resources', 1)
        elif slice_id == 2:
            #self.read_parameter_db('resources', 2)
            self.resources_2['MEC_CPU'] -= request['SLICE_MEC_CPU_REQUEST']
            self.resources_2['MEC_RAM'] -= request['SLICE_MEC_RAM_REQUEST']
            self.resources_2['MEC_STORAGE'] -= request['SLICE_MEC_STORAGE_REQUEST']
            self.resources_2['MEC_BW'] -= request['SLICE_MEC_BW_REQUEST']
            #self.update_db('resources', 2)
        elif slice_id == 3:
            #self.read_parameter_db('resources', 3)
            self.resources_3['MEC_CPU'] -= request['SLICE_MEC_CPU_REQUEST']
            self.resources_3['MEC_RAM'] -= request['SLICE_MEC_RAM_REQUEST']
            self.resources_3['MEC_STORAGE'] -= request['SLICE_MEC_STORAGE_REQUEST']
            self.resources_3['MEC_BW'] -= request['SLICE_MEC_BW_REQUEST']
            #self.update_db('resources', 3)
        elif slice_id == 4:
            #self.read_parameter_db('resources', 4)
            self.resources_4['MEC_CPU'] -= request['SLICE_MEC_CPU_REQUEST']
            self.resources_4['MEC_RAM'] -= request['SLICE_MEC_RAM_REQUEST']
            self.resources_4['MEC_STORAGE'] -= request['SLICE_MEC_STORAGE_REQUEST']
            self.resources_4['MEC_BW'] -= request['SLICE_MEC_BW_REQUEST']
            #self.update_db('resources', 4)
        elif slice_id == 5:
            #self.read_parameter_db('resources', 5)
            self.resources_5['MEC_CPU'] -= request['SLICE_MEC_CPU_REQUEST']
            self.resources_5['MEC_RAM'] -= request['SLICE_MEC_RAM_REQUEST']
            self.resources_5['MEC_STORAGE'] -= request['SLICE_MEC_STORAGE_REQUEST']
            self.resources_5['MEC_BW'] -= request['SLICE_MEC_BW_REQUEST']
            #self.update_db('resources', 5)
        elif slice_id == 6:
            #self.read_parameter_db('resources', 6)
            self.resources_6['MEC_CPU'] -= request['SLICE_MEC_CPU_REQUEST']
            self.resources_6['MEC_RAM'] -= request['SLICE_MEC_RAM_REQUEST']
            self.resources_6['MEC_STORAGE'] -= request['SLICE_MEC_STORAGE_REQUEST']
            self.resources_6['MEC_BW'] -= request['SLICE_MEC_BW_REQUEST']
            #self.update_db('resources', 6)
    
    def deallocate_slice(self, request, slice_id):
        # Function to deallocate resources of killed requests

        # RAN Resources Part-----------------------------------------------------------
        if request['SLICE_RAN_L_REQUEST'] > 10:
            #self.read_parameter_db('PRB_map1', 0)
            indices = np.where(self.PRB_map1 == request['UE_ID'])

            for i in range(len(indices[0])):
                #print(f"({indices[0][i]}, {indices[1][i]})")
                self.PRB_map1[indices[0][i], indices[1][i]] = 0
            #self.update_db('PRB_map1', 0)
        else:
            #self.read_parameter_db('PRB_map2', 0)
            indices = np.where(self.PRB_map2 == request['UE_ID'])

            for i in range(len(indices[0])):
                #print(f"({indices[0][i]}, {indices[1][i]})")
                self.PRB_map2[indices[0][i], indices[1][i]] = 0
            #self.update_db('PRB_map2', 0)

        # MEC Resources Part------------------------------------------------------------
        if slice_id == 1:
            #self.read_parameter_db('resources', 1)
            self.resources_1['MEC_CPU'] += request['SLICE_MEC_CPU_REQUEST']
            self.resources_1['MEC_RAM'] += request['SLICE_MEC_RAM_REQUEST']
            self.resources_1['MEC_STORAGE'] += request['SLICE_MEC_STORAGE_REQUEST']
            self.resources_1['MEC_BW'] += request['SLICE_MEC_BW_REQUEST']
            #self.update_db('resources', 1)
        elif slice_id == 2:
            #self.read_parameter_db('resources', 2)
            self.resources_2['MEC_CPU'] += request['SLICE_MEC_CPU_REQUEST']
            self.resources_2['MEC_RAM'] += request['SLICE_MEC_RAM_REQUEST']
            self.resources_2['MEC_STORAGE'] += request['SLICE_MEC_STORAGE_REQUEST']
            self.resources_2['MEC_BW'] += request['SLICE_MEC_BW_REQUEST']
            #self.update_db('resources', 2)
        elif slice_id == 3:
            #self.read_parameter_db('resources', 3)
            self.resources_3['MEC_CPU'] += request['SLICE_MEC_CPU_REQUEST']
            self.resources_3['MEC_RAM'] += request['SLICE_MEC_RAM_REQUEST']
            self.resources_3['MEC_STORAGE'] += request['SLICE_MEC_STORAGE_REQUEST']
            self.resources_3['MEC_BW'] += request['SLICE_MEC_BW_REQUEST']
            #self.update_db('resources', 3)
        elif slice_id == 4:
            #self.read_parameter_db('resources', 4)
            self.resources_4['MEC_CPU'] += request['SLICE_MEC_CPU_REQUEST']
            self.resources_4['MEC_RAM'] += request['SLICE_MEC_RAM_REQUEST']
            self.resources_4['MEC_STORAGE'] += request['SLICE_MEC_STORAGE_REQUEST']
            self.resources_4['MEC_BW'] += request['SLICE_MEC_BW_REQUEST']
            #self.update_db('resources', 4)
        elif slice_id == 5:
            #self.read_parameter_db('resources', 5)
            self.resources_5['MEC_CPU'] += request['SLICE_MEC_CPU_REQUEST']
            self.resources_5['MEC_RAM'] += request['SLICE_MEC_RAM_REQUEST']
            self.resources_5['MEC_STORAGE'] += request['SLICE_MEC_STORAGE_REQUEST']
            self.resources_5['MEC_BW'] += request['SLICE_MEC_BW_REQUEST']
            #self.update_db('resources', 5)
        elif slice_id == 6:
            #self.read_parameter_db('resources', 6)
            self.resources_6['MEC_CPU'] += request['SLICE_MEC_CPU_REQUEST']
            self.resources_6['MEC_RAM'] += request['SLICE_MEC_RAM_REQUEST']
            self.resources_6['MEC_STORAGE'] += request['SLICE_MEC_STORAGE_REQUEST']
            self.resources_6['MEC_BW'] += request['SLICE_MEC_BW_REQUEST']
            #self.update_db('resources', 6)
        
    def create_slice (self, request):
        # Function to create the slice for a specific request
        # This function inserts the defined slice to the request in the processed requests list
        #         self.slices_param = {1: [4, 16, 100, 40, 50], 2: [4, 32, 100, 100, 30], 3: [8, 16, 32, 80, 20], 
        #                               4: [4, 8, 16, 50, 25], 5: [2, 8, 32, 40, 10], 6: [2, 8, 32, 40, 5]}
        slice1 = self.slices_param[1]
        slice2 = self.slices_param[2]
        slice3 = self.slices_param[3]
        slice4 = self.slices_param[4]
        slice5 = self.slices_param[5]
        slice6 = self.slices_param[6]
        
        if (request['SLICE_RAN_R_REQUEST'] >= slice1[4]):
            slice_id = 1
        elif (request['SLICE_RAN_R_REQUEST'] >= slice2[4]):
            slice_id = 2
        elif (request['SLICE_RAN_R_REQUEST'] >= slice4[4]):
            slice_id = 4
        elif (request['SLICE_RAN_R_REQUEST'] >= slice3[4]):
            slice_id = 3
        elif (request['SLICE_RAN_R_REQUEST'] >= slice5[4]):
            slice_id = 5
        elif (request['SLICE_RAN_R_REQUEST'] >= slice6[4]):
            slice_id = 6
        return slice_id

    def reset_resources(self):
        #self.resources_1 = {'MEC_CPU': 30, 'MEC_RAM': 128, 'MEC_STORAGE': 100, 'MEC_BW': 300}
        #self.resources_2 = {'MEC_CPU': 30, 'MEC_RAM': 128, 'MEC_STORAGE': 100, 'MEC_BW': 200}
        #self.resources_3 = {'MEC_CPU': 50, 'MEC_RAM': 128, 'MEC_STORAGE': 100, 'MEC_BW': 200}
        #self.resources_4 = {'MEC_CPU': 30, 'MEC_RAM': 128, 'MEC_STORAGE': 100, 'MEC_BW': 200}
        #self.resources_5 = {'MEC_CPU': 20, 'MEC_RAM': 64, 'MEC_STORAGE': 80, 'MEC_BW': 100}
        #self.resources_6 = {'MEC_CPU': 20, 'MEC_RAM': 64, 'MEC_STORAGE': 80, 'MEC_BW': 80}

        self.PRB_map1 = np.zeros((14, self.PRB_per_channel1))
        #self.update_db('PRB_map1', 0)

        self.PRB_map2 = np.zeros((14, self.PRB_per_channel2))
        #self.update_db('PRB_map2', 0)

        self.resources_1['MEC_CPU'] = 30
        self.resources_1['MEC_RAM'] = 128
        self.resources_1['MEC_STORAGE'] = 1000
        self.resources_1['MEC_BW'] = 300
        #self.update_db('resources', 1)

        self.resources_2['MEC_CPU'] = 30
        self.resources_2['MEC_RAM'] = 128
        self.resources_2['MEC_STORAGE'] = 1000
        self.resources_2['MEC_BW'] = 200
        #self.update_db('resources', 2)

        self.resources_3['MEC_CPU'] = 50
        self.resources_3['MEC_RAM'] = 128
        self.resources_3['MEC_STORAGE'] = 1000
        self.resources_3['MEC_BW'] = 200
        #self.update_db('resources', 3)

        self.resources_4['MEC_CPU'] = 30
        self.resources_4['MEC_RAM'] = 128
        self.resources_4['MEC_STORAGE'] = 1000
        self.resources_4['MEC_BW'] = 200
        #self.update_db('resources', 4)

        self.resources_5['MEC_CPU'] = 20
        self.resources_5['MEC_RAM'] = 64
        self.resources_5['MEC_STORAGE'] = 800
        self.resources_5['MEC_BW'] = 100
        #self.update_db('resources', 5)

        self.resources_6['MEC_CPU'] = 20
        self.resources_6['MEC_RAM'] = 64
        self.resources_6['MEC_STORAGE'] = 800
        self.resources_6['MEC_BW'] = 80
        #self.update_db('resources', 6)
    
    def evaluate_action(self, action, slice_id, reward_value, terminated):
        if action == 1 and slice_id == 1:
            self.check_resources(self.next_request)
            if self.resources_flag == 1:
                self.allocate_slice(self.next_request,slice_id)
                self.processed_requests[len(self.processed_requests) - 1]['SliceID'] = slice_id
                #self.update_db('processed_requests', 0)
                self.reward += reward_value   
                self.next_request = self.read_request()
            else: 
                terminated = True
                self.reward = 0
        
        if action == 1 and slice_id != 1:
            terminated = True
            self.reward = 0
            
        if action == 2 and slice_id == 2:
            self.check_resources(self.next_request)
            if self.resources_flag == 1:
                self.allocate_slice(self.next_request, slice_id)
                self.processed_requests[len(self.processed_requests) - 1]['SliceID'] = slice_id
                #self.update_db('processed_requests', 0)
                self.reward += reward_value   
                self.next_request = self.read_request()
            else: 
                terminated = True
                self.reward = 0
        
        if action == 2 and slice_id != 2:
            terminated = True
            self.reward = 0
            
        if action == 3 and slice_id == 3:
            self.check_resources(self.next_request)
            if self.resources_flag == 1:
                self.allocate_slice(self.next_request, slice_id)
                self.processed_requests[len(self.processed_requests) - 1]['SliceID'] = slice_id
                #self.update_db('processed_requests', 0)
                self.reward += reward_value   
                self.next_request = self.read_request()
            else: 
                terminated = True
                self.reward = 0
        
        if action == 3 and slice_id != 3:
            terminated = True
            self.reward = 0

        if action == 4 and slice_id == 4:
            self.check_resources(self.next_request)
            if self.resources_flag == 1:
                self.allocate_slice(self.next_request, slice_id)
                self.processed_requests[len(self.processed_requests) - 1]['SliceID'] = slice_id
                #self.update_db('processed_requests', 0)
                self.reward += reward_value   
                self.next_request = self.read_request()
            else: 
                terminated = True
                self.reward = 0
        
        if action == 4 and slice_id != 4:
            terminated = True
            self.reward = 0
            
        if action == 5 and slice_id == 5:
            self.check_resources(self.next_request)
            if self.resources_flag == 1:
                self.allocate_slice(self.next_request, slice_id)
                self.processed_requests[len(self.processed_requests) - 1]['SliceID'] = slice_id
                #self.update_db('processed_requests', 0)
                self.reward += reward_value   
                self.next_request = self.read_request()
            else: 
                terminated = True
                self.reward = 0
        
        if action == 5 and slice_id != 5:
            terminated = True
            self.reward = 0

        if action == 6 and slice_id == 6:
            self.check_resources(self.next_request)
            if self.resources_flag == 1:
                self.allocate_slice(self.next_request, slice_id)
                self.processed_requests[len(self.processed_requests) - 1]['SliceID'] = slice_id
                #self.update_db('processed_requests', 0)
                self.reward += reward_value   
                self.next_request = self.read_request()
            else: 
                terminated = True
                self.reward = 0
        
        if action == 6 and slice_id != 6:
            terminated = True
            self.reward = 0

        if action == 0:
            self.check_resources(self.next_request)
            if self.resources_flag == 0:
                self.reward += reward_value
                self.processed_requests.remove(self.processed_requests[len(self.processed_requests) - 1])
                self.next_request = self.read_request()
            else: 
                terminated = True
                self.reward = 0        
        
        return terminated

    def check_RAN(self, request):
        # Function to check whether are available resources or not at RAN level. It checks the correspondetn BWP according to the latency request (Greater than 10 ms is assigned to BWP1).

        if request['SLICE_RAN_L_REQUEST'] > 10:
            #self.read_parameter_db('PRB_map1', 0)
            indices = np.where(self.PRB_map1 == 0)
            available_symbols = len(indices[0])

            W_total = self.PRB_BW1* self.spectral_efficiency1 * available_symbols

            if request['SLICE_RAN_R_REQUEST'] * (10**6) <= W_total * log2(1 + request['UE_SiNR']):
                return True
            else: return False
        else:
            #self.read_parameter_db('PRB_map2', 0)
            indices = np.where(self.PRB_map2 == 0)
            available_symbols = len(indices[0])

            W_total = self.PRB_BW2 * self.spectral_efficiency2 * available_symbols

            if request['SLICE_RAN_R_REQUEST'] * (10**6) <= W_total * log2(1 + request['UE_SiNR']):
                return True
            else: return False

    def allocate_ran(self, request):

        if request['SLICE_RAN_L_REQUEST'] > 10:
            #self.read_parameter_db('PRB_map1', 0)
            indices = np.where(self.PRB_map1 == 0)

            number_symbols = ceil((request['SLICE_RAN_R_REQUEST'] * (10**6)) / (self.PRB_BW1 * self.spectral_efficiency1 * log2(1 + request['UE_SiNR'])))

            for i in range(number_symbols):
                #print(f"({indices[0][i]}, {indices[1][i]})")
                self.PRB_map1[indices[0][i], indices[1][i]] = request['UE_ID']
            #self.update_db('PRB_map1', 0)
        else:
            #self.read_parameter_db('PRB_map2', 0)
            indices = np.where(self.PRB_map2 == 0)

            number_symbols = ceil((request['SLICE_RAN_R_REQUEST'] * (10**6)) / (self.PRB_BW2 * self.spectral_efficiency2 * log2(1 + request['UE_SiNR'])))

            for i in range(number_symbols):
                #print(f"({indices[0][i]}, {indices[1][i]})")
                self.PRB_map2[indices[0][i], indices[1][i]] = request['UE_ID']
            #self.update_db('PRB_map2', 0)
        
#a = SliceCreationEnv5()
#check_env(a)