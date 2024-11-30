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


#------------------------------------Environment Class----------------------------------------------------------------------------
class SliceManagementEnv1(gym.Env):
    metadata = {"render_modes": [], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        # Define environment parameters

        self.current_episode = 1
        self.sample_size = 4
        self.buffer_requests = {}
        
        #Uncomment to train
        #self.select_db = randint(1,500)

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

        #Defined parameters per Slice. (Each component is a list of the correspondent slice parameters)-------------------------------------------------------------------------------------
        self.slices_param = {1: [4, 16, 100, 40, 50, 20], 2: [4, 32, 100, 100, 30, 30], 3: [8, 16, 32, 80, 20, 1], 
                             4: [4, 8, 16, 50, 25, 5], 5: [2, 8, 32, 40, 10, 10], 6: [2, 8, 32, 40, 5, 40]}
        
        # VECTORS----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.observation_space = gym.spaces.Box(low=0, high=10000, shape=(2,), dtype=np.float32) #ovservation space composed by Requested resources (MEC BW) and available MEC resources.
        
        self.action_space = gym.spaces.Discrete(3)  # 0: Blocking, 1: Execute Configuration Action , 2: Do Nothing
        
        # Other necessary variables and data structures
        self.current_time_step = 1
        self.reward = 0
        self.first = True

        self.maintain_request = 0
        
        # Flags -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.config_flag = 0
        self.resources_flag = 1
        

        #--------------------------------------------------------------------------------------Imported variables ---------------------------------------------------------------------------------
        self.processed_requests = []
        #self.read_parameter_db('processed_requests', 0)

        #self.PRB_map = np.zeros((14, self.PRB_per_channel))                                 # PRB map per slot (14 OFDM symbols x Number of PRB to allocate within the channel bandwidth)
        #self.read_parameter_db('PRB_map1', 0)
        #self.read_parameter_db('PRB_map2', 0)


        #Available MEC resources (Order: MEC_CPU (Cores), MEC_RAM (GB), MEC_STORAGE (GB), MEC_BW (Mbps))
        #self.resources = [1000]
        self.resources_1 = {'MEC_CPU': 30, 'MEC_RAM': 128, 'MEC_STORAGE': 100, 'MEC_BW': 300}
        #self.read_parameter_db('resources', 1)
        self.resources_2 = {'MEC_CPU': 30, 'MEC_RAM': 128, 'MEC_STORAGE': 100, 'MEC_BW': 200}
        #self.read_parameter_db('resources', 2)
        self.resources_3 = {'MEC_CPU': 50, 'MEC_RAM': 128, 'MEC_STORAGE': 100, 'MEC_BW': 200}
        #self.read_parameter_db('resources', 3)
        self.resources_4 = {'MEC_CPU': 30, 'MEC_RAM': 128, 'MEC_STORAGE': 100, 'MEC_BW': 200}
        #self.read_parameter_db('resources', 4)
        self.resources_5 = {'MEC_CPU': 20, 'MEC_RAM': 64, 'MEC_STORAGE': 80, 'MEC_BW': 100}
        #self.read_parameter_db('resources', 5)
        self.resources_6 = {'MEC_CPU': 20, 'MEC_RAM': 64, 'MEC_STORAGE': 80, 'MEC_BW': 80}
        #self.read_parameter_db('resources', 6)

    #-------------------------Reset Method----------------------------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        # Initialize the environment to its initial state

        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        #Uncomment to train
        #self.select_db = randint(1,500)

        self.reward = 0
        
        #self.simulate_noise()
        self.read_parameter_db('processed_requests', 0)

        self.reset_resources()

        #self.update_slice_requests(self.next_request)

        self.config_flag = 0
        self.resources_flag = 1

        self.maintain_request = 0

        self.check_resources()

        self.observation = np.array([self.config_flag] + [self.resources_flag], dtype=np.float32)

        self.info = {}
        self.first = True
        
        #print("\nReset: ", self.observation)

        self.current_episode += 1
        
        return self.observation, self.info

    #-------------------------Step Method-----------------------------------------------------------------------------------------
    def step(self, action):
            
        terminated = False
        
        reward_value = 1
    
        # Apply the selected action (0: Do Nothing, 1: Allocate Slice 1, 2: Allocate Slice 2, 3: Allocate Slice 3)
        terminated = self.evaluate_action(action, reward_value, terminated) 

        self.simulate_noise()
        self.read_parameter_db('processed_requests', 0)
        self.read_parameter_db('PRB_map1', 0)
        self.read_parameter_db('PRB_map2', 0)

        self.check_resources()
    
        #self.observation = np.array([self.next_request[1]] + [self.resources_flag], dtype=np.float32)
        self.observation = np.array([self.config_flag]+ [self.resources_flag], dtype=np.float32)
        
        #done = False
        
        info = {}  # Additional information (if needed)
        
        self.current_time_step += 1  # Increment the time step
        
        #print("Action: ", action, "\nObservation: ", self.observation, "\nReward: ", self.reward)

        truncated = False
        
        return self.observation, self.reward, terminated, truncated, info

    #------------------------Other Functions--------------------------------------------------------------------------------------
    def check_maintain(self):
        # Function to check whether a configuration action is needed or not
        self.read_parameter_db('processed_requests', 0)
        self.read_parameter_db('PRB_map1', 0)
        self.read_parameter_db('PRB_map2', 0)

        for i in self.processed_requests[:-1]:

            if i['SLICE_RAN_L_REQUEST'] > 10:
                indices = np.where(self.PRB_map1 == i['UE_ID'])
                allocated_symbols = len(indices[0])

                needed_symbols = ceil((i['SLICE_RAN_R_REQUEST'] * (10**6)) / (self.PRB_BW1 * self.spectral_efficiency1 * log2(1 + i['UE_SiNR'])))
            else:
                indices = np.where(self.PRB_map2 == i['UE_ID'])
                allocated_symbols = len(indices[0])

                needed_symbols = ceil((i['SLICE_RAN_R_REQUEST'] * (10**6)) / (self.PRB_BW2 * self.spectral_efficiency2 * log2(1 + i['UE_SiNR'])))

            if allocated_symbols < needed_symbols: 
                if str(i['UE_ID']) not in self.buffer_requests.keys():
                    self.buffer_requests[str(i['UE_ID'])] = [i['SLICE_RAN_R_REQUEST']]
                else:
                    self.buffer_requests[str(i['UE_ID'])].append(i['SLICE_RAN_R_REQUEST'])
                
                if len(self.buffer_requests[str(i['UE_ID'])]) == self.sample_size:
                    #sample_R = self.buffer_requests[str(i['UE_ID'])][-1]
                    #needed_symbols = ceil((sample_R * (10**6)) / (self.PRB_BW2 * self.spectral_efficiency2 * log2(1 + i['UE_SiNR'])))
                    #self.buffer_requests[str(i['UE_ID'])] = []
                    if allocated_symbols < needed_symbols:
                        self.config_flag = 1
                        self.maintain_request = i['UE_ID']
                        break
                    else: self.config_flag = 0
                else: self.config_flag = 0
            else: self.config_flag = 0

            '''
            if allocated_symbols < needed_symbols: 
                if i['UE_ID'] not in self.buffer_requests:
                    self.buffer_requests['{}'.format(i['UE_ID'])] = [i['SLICE_RAN_R_REQUEST']]
                else:
                    self.buffer_requests['{}'.format(i['UE_ID'])].append(i['SLICE_RAN_R_REQUEST'])
                
                if len(self.buffer_requests['{}'.format(i['UE_ID'])]) == self.sample_size:
                    sample_R = self.buffer_requests['{}'.format(i['UE_ID'])][-1]
                    needed_symbols = ceil((sample_R * (10**6)) / (self.PRB_BW2 * self.spectral_efficiency2 * log2(1 + i['UE_SiNR'])))
                    self.buffer_requests['{}'.format(i['UE_ID'])] = []
                    if allocated_symbols < needed_symbols:
                        self.config_flag = 1
                        self.maintain_request = i['UE_ID']
                        break
                    else: self.config_flag = 0
                else: self.config_flag = 0
            else: self.config_flag = 0'''

    def check_resources(self):
        
        self.check_maintain()

        if self.maintain_request != 0:
            request = next((d for d in self.processed_requests[:-1] if d.get('UE_ID') == self.maintain_request), None)  # Obtain the request that needs to be guaranteed

            #Check RAN Resources ----------------------------------------------------------------------------------------------------------------------------
            self.check_RAN(request)

            # Check MEC resources----------------------------------------------------------------------------------------------------------------------------
            '''
            if slice_id == 1:
                self.read_parameter_db('resources', 1)
                if (self.resources_1['MEC_CPU'] >= request['SLICE_MEC_CPU_REQUEST'] and self.resources_1['MEC_RAM'] >= request['SLICE_MEC_RAM_REQUEST'] and 
                    self.resources_1['MEC_STORAGE'] >= request['SLICE_MEC_STORAGE_REQUEST'] and self.resources_1['MEC_BW'] >= request['SLICE_MEC_BW_REQUEST'] and ran_resources
                    ):
                    self.resources_flag = 1
                else: self.resources_flag = 0
            elif slice_id == 2:
                self.read_parameter_db('resources', 2)
                if (self.resources_2['MEC_CPU'] >= request['SLICE_MEC_CPU_REQUEST'] and self.resources_2['MEC_RAM'] >= request['SLICE_MEC_RAM_REQUEST'] and 
                    self.resources_2['MEC_STORAGE'] >= request['SLICE_MEC_STORAGE_REQUEST'] and self.resources_2['MEC_BW'] >= request['SLICE_MEC_BW_REQUEST'] and ran_resources
                    ):
                    self.resources_flag = 1
                else: self.resources_flag = 0
            elif slice_id == 3:
                self.read_parameter_db('resources', 3)
                if (self.resources_3['MEC_CPU'] >= request['SLICE_MEC_CPU_REQUEST'] and self.resources_3['MEC_RAM'] >= request['SLICE_MEC_RAM_REQUEST'] and 
                    self.resources_3['MEC_STORAGE'] >= request['SLICE_MEC_STORAGE_REQUEST'] and self.resources_3['MEC_BW'] >= request['SLICE_MEC_BW_REQUEST'] and ran_resources
                    ):
                    self.resources_flag = 1
                else: self.resources_flag = 0
            elif slice_id == 4:
                self.read_parameter_db('resources', 4)
                if (self.resources_4['MEC_CPU'] >= request['SLICE_MEC_CPU_REQUEST'] and self.resources_4['MEC_RAM'] >= request['SLICE_MEC_RAM_REQUEST'] and 
                    self.resources_4['MEC_STORAGE'] >= request['SLICE_MEC_STORAGE_REQUEST'] and self.resources_4['MEC_BW'] >= request['SLICE_MEC_BW_REQUEST'] and ran_resources
                    ):
                    self.resources_flag = 1
                else: self.resources_flag = 0
            elif slice_id == 5:
                self.read_parameter_db('resources', 5)
                if (self.resources_5['MEC_CPU'] >= request['SLICE_MEC_CPU_REQUEST'] and self.resources_5['MEC_RAM'] >= request['SLICE_MEC_RAM_REQUEST'] and 
                    self.resources_5['MEC_STORAGE'] >= request['SLICE_MEC_STORAGE_REQUEST'] and self.resources_5['MEC_BW'] >= request['SLICE_MEC_BW_REQUEST'] and ran_resources
                    ):
                    self.resources_flag = 1
                else: self.resources_flag = 0
            elif slice_id == 6:
                self.read_parameter_db('resources', 6)
                if (self.resources_6['MEC_CPU'] >= request['SLICE_MEC_CPU_REQUEST'] and self.resources_6['MEC_RAM'] >= request['SLICE_MEC_RAM_REQUEST'] and 
                    self.resources_6['MEC_STORAGE'] >= request['SLICE_MEC_STORAGE_REQUEST'] and self.resources_6['MEC_BW'] >= request['SLICE_MEC_BW_REQUEST'] and ran_resources
                    ):
                    self.resources_flag = 1
                else: self.resources_flag = 0 '''
         
    def reset_resources(self):
        self.read_parameter_db('PRB_map1', 0)
        self.read_parameter_db('PRB_map2', 0)

        #Available MEC resources (Order: MEC_CPU (Cores), MEC_RAM (GB), MEC_STORAGE (GB), MEC_BW (Mbps))
        self.read_parameter_db('resources', 1)
        self.read_parameter_db('resources', 2)
        self.read_parameter_db('resources', 3)
        self.read_parameter_db('resources', 4)
        self.read_parameter_db('resources', 5)
        self.read_parameter_db('resources', 6)
    
    def evaluate_action(self, action, reward_value, terminated):
        if action == 1:
            #self.check_resources()
            if self.resources_flag == 1 and self.config_flag == 1:
                for d in self.processed_requests[:-1]:
                    if d.get('UE_ID') == self.maintain_request:
                        self.allocate_ran(d)
                        break
                self.update_db('PRB_map1', 0)
                self.update_db('PRB_map2', 0)
                self.update_db('processed_requests', 0)
                self.reward += reward_value   
            else: 
                terminated = True
                self.reward = 0
            
        if action == 2:
            #self.check_resources()
            if self.resources_flag == 1 and self.config_flag == 0:
                self.reward += reward_value   
            else: 
                terminated = True
                self.reward = 0

        if action == 0:
            #self.check_resources()
            if self.resources_flag == 0:
                self.reward += reward_value
            else: 
                terminated = True
                self.reward = 0        
        
        return terminated

    def check_RAN(self, request):

        if request['SLICE_RAN_L_REQUEST'] > 10:
            indices = np.where(self.PRB_map1 == 0)
            available_symbols = len(indices[0])

            indices_a = np.where(self.PRB_map1 == request['UE_ID'])
            allocated_symbols = len(indices_a[0])

            W_total = self.PRB_BW1 * self.spectral_efficiency1 * (available_symbols + allocated_symbols)

            if request['SLICE_RAN_R_REQUEST'] * (10**6) <= W_total * log2(1 + request['UE_SiNR']):
                self.resources_flag = 1
            else: self.resources_flag = 0
        else:
            indices = np.where(self.PRB_map2 == 0)
            available_symbols = len(indices[0])

            indices_a = np.where(self.PRB_map2 == request['UE_ID'])
            allocated_symbols = len(indices_a[0])

            W_total = self.PRB_BW2 * self.spectral_efficiency2 * (available_symbols + allocated_symbols)

            if request['SLICE_RAN_R_REQUEST'] * (10**6) <= W_total * log2(1 + request['UE_SiNR']):
                self.resources_flag = 1
            else: self.resources_flag = 0

    def allocate_ran(self, request):

        if request['SLICE_RAN_L_REQUEST'] > 10:
            indices = np.where(self.PRB_map1 == 0)
            indices_allocated = np.where(self.PRB_map1 == request['UE_ID'])

            number_symbols = ceil((request['SLICE_RAN_R_REQUEST'] * (10**6)) / (self.PRB_BW1 * self.spectral_efficiency1 * log2(1 + request['UE_SiNR'])))

            for i in range(number_symbols - len(indices_allocated[0])):
                #print(f"({indices[0][i]}, {indices[1][i]})")
                self.PRB_map1[indices[0][i], indices[1][i]] = request['UE_ID']
            self.update_db('PRB_map1', 0)
            self.update_db('PRB_map2', 0)
        else:
            indices = np.where(self.PRB_map2 == 0)
            indices_allocated = np.where(self.PRB_map2 == request['UE_ID'])

            number_symbols = ceil((request['SLICE_RAN_R_REQUEST'] * (10**6)) / (self.PRB_BW2 * self.spectral_efficiency2 * log2(1 + request['UE_SiNR'])))

            for i in range(number_symbols - len(indices_allocated[0])):
                #print(f"({indices[0][i]}, {indices[1][i]})")
                self.PRB_map2[indices[0][i], indices[1][i]] = request['UE_ID']
            self.update_db('PRB_map1', 0)
            self.update_db('PRB_map2', 0)

    def read_parameter_db(self, parameter, number):
        # Connect to the SQLite database
        #conn = sqlite3.connect('data/Global_Parameters{}.db'.format(str(self.select_db)))
        conn = sqlite3.connect('/home/mario/Documents/DQN_Models/Joint/Global_Parameters.db')  
        cursor = conn.cursor()

        if parameter == 'processed_requests':

            # Query the database to retrieve stored data
            cursor.execute('''SELECT processed_requests FROM Parameters''')
            row = cursor.fetchone()
            self.processed_requests = json.loads(row[0])

        if parameter == 'PRB_map1':

            # Query the database to retrieve stored data
            cursor.execute('''SELECT PRB_map1 FROM Parameters''')
            row = cursor.fetchone()
            self.PRB_map1 = np.frombuffer(bytearray(row[0]), dtype=np.int64).reshape((14, self.PRB_per_channel1))

        if parameter == 'PRB_map2':

            # Query the database to retrieve stored data
            cursor.execute('''SELECT PRB_map2 FROM Parameters''')
            row = cursor.fetchone()
            self.PRB_map2 = np.frombuffer(bytearray(row[0]), dtype=np.int64).reshape((14, self.PRB_per_channel2))

        if parameter == 'resources':
            # Query the database to retrieve stored data
            cursor.execute('''SELECT resources_{} FROM Parameters'''.format(str(number)))
            row = cursor.fetchone()

            match number:
                case 1:
                    self.resources_1 = json.loads(row[0])
                case 2:
                    self.resources_2 = json.loads(row[0])
                case 3:
                    self.resources_3 = json.loads(row[0])
                case 4:
                    self.resources_4 = json.loads(row[0])
                case 5: 
                    self.resources_5 = json.loads(row[0])
                case 6:
                    self.resources_6 = json.loads(row[0])

        # Commit changes and close connection
        conn.commit()

        # Close connection
        conn.close()

    def update_db(self, parameter, number):
        # Connect to the SQLite database
        #conn = sqlite3.connect('data/Global_Parameters{}.db'.format(str(self.select_db)))
        conn = sqlite3.connect('/home/mario/Documents/DQN_Models/Joint/Global_Parameters.db')
        cursor = conn.cursor()

        if parameter == 'processed_requests':
            # Serialize data
            serialized_parameter = json.dumps(self.processed_requests)
            #print(serialized_parameter)

            cursor.execute('''UPDATE Parameters SET processed_requests = ? WHERE rowid = 1''', (serialized_parameter,))

        if parameter == 'PRB_map1':
            # Serialize data
            serialized_parameter = self.PRB_map1.tobytes()

            cursor.execute('''UPDATE Parameters SET PRB_map1 = ? WHERE rowid = 1''', (serialized_parameter,)) 

        if parameter == 'PRB_map2':
            # Serialize data
            serialized_parameter = self.PRB_map2.tobytes()

            cursor.execute('''UPDATE Parameters SET PRB_map2 = ? WHERE rowid = 1''', (serialized_parameter,)) 

        if parameter == 'resources':
            match number:
                case 1:
                    #Serialize data
                    serialized_parameter = json.dumps(self.resources_1)
                case 2:
                    #Serialize data
                    serialized_parameter = json.dumps(self.resources_2)
                case 3:
                    #Serialize data
                    serialized_parameter = json.dumps(self.resources_3)
                case 4:
                    #Serialize data
                    serialized_parameter = json.dumps(self.resources_4)
                case 5:
                     #Serialize data
                    serialized_parameter = json.dumps(self.resources_5)
                case 6:
                     #Serialize data
                    serialized_parameter = json.dumps(self.resources_6)

            cursor.execute('''UPDATE Parameters SET resources_{} = ? WHERE rowid = 1'''.format(str(number)), (serialized_parameter,))

        # Commit changes and close connection
        conn.commit()
        conn.close()

    def simulate_noise(self):
        self.read_parameter_db('processed_requests', 0)

        '''
        if len(self.processed_requests) >= 3:
            index_request = randint(0, (len(self.processed_requests)-2))
            self.processed_requests[index_request]['UE_SiNR'] = randint(1, 20)

            self.update_db('processed_requests', 0)'''
        
        if self.current_time_step == 50:
            index = next((i for i, d in enumerate(self.processed_requests) if d.get('UE_ID') == 3), None)
            if index != None:
                self.processed_requests[index]['UE_SiNR'] = 4
                self.update_db('processed_requests', 0)
            
#a = SliceManagementEnv1()
#check_env(a)