import gymnasium as gym

from stable_baselines3 import DQN

from gym_examples.envs.slice_creation_env5 import SliceCreationEnv5
from gym_examples.envs.slice_management_env1 import SliceManagementEnv1

from os import rename
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def calculate_utilization_mec(parameter, current, total):
    
    utilization = ((total - current) / total) * 100
    
    match parameter:
        case 'cpu':
            mec_cpu_utilization.append(utilization)
        case 'ram':
            mec_ram_utilization.append(utilization)
        case 'storage':
            mec_storage_utilization.append(utilization)
        case 'bw':
            mec_bw_utilization.append(utilization)

def calculate_utilization_ran(bwp, current):

    indices = np.where(current == 0)
    available_symbols = len(indices[0])

    utilization = ((current.size - available_symbols) / current.size) * 100

    if bwp == 'bwp1':
        ran_bwp1_utilization.append(utilization)
    
    if bwp == 'bwp2':
        ran_bwp2_utilization.append(utilization)

def find_index(dicts, key, value):
    return next((i for i, d in enumerate(dicts) if d.get(key) == value), -1)

env1 = SliceCreationEnv5()
env2 = SliceManagementEnv1()


model1 = DQN.load("/home/mario/Documents/DQN_Models/Joint/gym-examples5/dqn_slices1_1005.zip", env1)
#model2 = DQN.load("/home/mario/Documents/DQN_Models/Joint/gym-examples1/dqn_maintain1_0905.zip", env2)
model2 = DQN.load("/home/mario/Documents/DQN_Models/Joint/gym-examples5/dqn_maintain1.zip", env2)

obs1, info1 = env1.reset()
obs2, info2 = env2.reset()

cont = 0
cont_rejections = 0
mec_cpu_utilization = []
mec_ram_utilization = []
mec_storage_utilization = []
mec_bw_utilization = []
ran_bwp1_utilization = []
ran_bwp2_utilization = []

request_example = []
prb_example1 = []
tstep_example1 = []
prb_example2 = []
tstep_example2 = []
snr_example1 = []
snr_example2 = []
'''
for i in range(500):
    while cont<99:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        print('Action: ', action,'Observation: ', obs, ' | Reward: ', reward, ' | Terminated: ', terminated)
        cont += 1
        if terminated or truncated:
            obs, info = env.reset()
    #cont = 0
    # Comment after training of Model 2
    #rename('Global_Parameters.db','Global_Parameters{}.db'.format(str(i+1)))
    #obs, info = env.reset()
    '''

while cont<500:
    action1, _states1 = model1.predict(obs1, deterministic=True)
    action2, _states2 = model2.predict(obs2, deterministic=True)

    if action1 == 0: cont_rejections += 1

    obs1, reward1, terminated1, truncated1, info1 = env1.step(action1)
    obs2, reward2, terminated2, truncated2, info2 = env2.step(action2)

    calculate_utilization_mec('cpu', env1.resources_1['MEC_CPU'], 128)
    calculate_utilization_mec('ram', env1.resources_1['MEC_RAM'], 512)
    calculate_utilization_mec('storage', env1.resources_1['MEC_STORAGE'], 5000)
    calculate_utilization_mec('bw', env1.resources_1['MEC_BW'], 2000)
    calculate_utilization_ran('bwp1', env1.PRB_map1)
    calculate_utilization_ran('bwp2', env1.PRB_map2)

    if len((np.where(env1.PRB_map1 == 3))[0]) != 0:
        index = find_index(env2.processed_requests, 'UE_ID', 3)
        request_example = env2.processed_requests[index]
        example_symbols1 = np.where(env1.PRB_map1 == 3)
        example_symbols1 = len(example_symbols1[0])
        #print(env2.processed_requests[2]['UE_SiNR'])
        snr_example1.append(next((d['UE_SiNR'] for d in env2.processed_requests[:-1] if d.get('UE_ID') == 3)), None)
        prb_example1.append(example_symbols1)
        tstep_example1.append(env2.current_time_step)

    if len((np.where(env1.PRB_map2 == 3))[0]) != 0:
        index = find_index(env2.processed_requests, 'UE_ID', 3)
        request_example = env2.processed_requests[index]
        example_symbols2 = np.where(env1.PRB_map2 == 3)
        example_symbols2 = len(example_symbols2[0])
        #print(env2.processed_requests[2]['UE_SiNR'])
        snr_example2.append(next((d['UE_SiNR'] for d in env2.processed_requests[:-1] if d.get('UE_ID') == 3), None))
        prb_example2.append(example_symbols2)
        tstep_example2.append(env2.current_time_step)

    print("Model 1: ",'Action: ', action1,'Observation: ', obs1, ' | Reward: ', reward1, ' | Terminated: ', terminated1)
    print("Model 2: ",'Action: ', action2,'Observation: ', obs2, ' | Reward: ', reward2, ' | Terminated: ', terminated2)

    cont += 1
    if terminated1 or truncated1:
        obs1, info1 = env1.reset()
    
    if terminated2 or truncated2:
        obs2, info2 = env2.reset()

x = np.linspace(0, len(mec_cpu_utilization), len(mec_cpu_utilization))
#print(len(x), len(mec_cpu_utilization))

print(cont_rejections)
print(len(prb_example1))
print(len(tstep_example1))
print(len(prb_example2))
print(len(tstep_example2))

plt.figure(1)
plt.plot(x, mec_cpu_utilization, marker='.', linestyle='-', color='r', label='MEC CPU')
plt.plot(x, mec_ram_utilization, marker='.', linestyle='-', color='b', label='MEC RAM')
plt.plot(x, mec_storage_utilization, marker='.', linestyle='-', color='g', label='MEC STORAGE')
plt.plot(x, mec_bw_utilization, marker='.', linestyle='-', color='y', label='MEC BW')
plt.ylim(0, 100)
plt.xlabel("Timesteps")
plt.ylabel("Utilization %")
plt.legend()

plt.figure(2)
plt.plot(x, ran_bwp1_utilization, marker='.', linestyle='-', color='c', label='BWP 1')
plt.plot(x, ran_bwp2_utilization, marker='.', linestyle='-', color='m', label='BWP 2')
plt.ylim(0, 100)
plt.xlabel("Timesteps")
plt.ylabel("Utilization %")
plt.legend()

plt.figure(3)
plt.subplot(211)
plt.plot(tstep_example1, prb_example1, marker='.', linestyle='-', color='k', label='Slice Request ID = 3')
plt.xlim(0, 500)
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel("Timesteps")
plt.ylabel("Assigned PRBs")
plt.legend()

if len(snr_example1) != 0:
    y = snr_example1
    x = tstep_example1
else:
    y = snr_example2
    x = tstep_example2

plt.subplot(212)
plt.plot(x, y, marker='.', linestyle='-', color='k', label='SiNR Request ID = 3')
plt.xlim(0, 500)
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel("Timesteps")
plt.ylabel("SiNR")
plt.legend()

plt.figure(4)
plt.subplot(211)
plt.plot(tstep_example2, prb_example2, marker='.', linestyle='-', color='k', label='Slice Request ID = 3')
plt.xlim(0, 500)
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel("Timesteps")
plt.ylabel("Assigned PRBs")
plt.legend()

if len(snr_example1) != 0:
    y = snr_example1
    x = tstep_example1
else:
    y = snr_example2
    x = tstep_example2

plt.subplot(212)
plt.plot(x, y, marker='.', linestyle='-', color='k', label='SiNR Request ID = 3')
plt.xlim(0, 500)
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel("Timesteps")
plt.ylabel("SiNR")
plt.legend()

plt.show()

print(request_example)