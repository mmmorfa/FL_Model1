import gymnasium as gym

from stable_baselines3 import DQN

from gym_examples.envs.slice_creation_env5 import SliceCreationEnv5
from gym_examples.envs.slice_management_env1 import SliceManagementEnv1

import numpy as np
import matplotlib.pyplot as plt
import torch, os

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

# Force PyTorch to use CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
device = torch.device("cpu")

model1 = DQN.load("Trained_Models/client_1_model_round_5.zip", env1)
model2 = DQN.load("Trained_Models/dqn_Model_2.zip", env2)

obs1, info1 = env1.reset()
obs2, info2 = env2.reset()

cont = 0

cont_rejections1 = 0
cont_rejections2 = 0

rewards_model1 = []
rewards_model2 = []

mec_cpu_utilization = []
mec_ram_utilization = []
mec_storage_utilization = []
mec_bw_utilization = []
ran_bwp1_utilization = []
ran_bwp2_utilization = []


while cont<100:
    action1, _states1 = model1.predict(obs1, deterministic=True)
    action2, _states2 = model2.predict(obs2, deterministic=True)

    if action1 == 0: cont_rejections1 += 1
    if action2 == 0: cont_rejections2 += 1

    obs1, reward1, terminated1, truncated1, info1 = env1.step(action1)
    obs2, reward2, terminated2, truncated2, info2 = env2.step(action2)

    rewards_model1.append(reward1)
    rewards_model2.append(reward2)

    calculate_utilization_mec('cpu', env1.resources_1['MEC_CPU'], 128)
    calculate_utilization_mec('ram', env1.resources_1['MEC_RAM'], 512)
    calculate_utilization_mec('storage', env1.resources_1['MEC_STORAGE'], 5000)
    calculate_utilization_mec('bw', env1.resources_1['MEC_BW'], 2000)
    calculate_utilization_ran('bwp1', env1.PRB_map1)
    calculate_utilization_ran('bwp2', env1.PRB_map2)

    print("Model 1: ",'Action: ', action1,'Observation: ', obs1, ' | Reward: ', reward1, ' | Terminated: ', terminated1)
    print("Model 2: ",'Action: ', action2,'Observation: ', obs2, ' | Reward: ', reward2, ' | Terminated: ', terminated2)



    cont += 1
    if terminated1 or truncated1:
        obs1, info1 = env1.reset()

    if terminated2 or truncated2:
        obs2, info2 = env2.reset()

# Example logs (replace these with parsed data from your logs)
evaluation_steps = list(range(1, 101))  # Assuming 100 evaluation steps

# Create a plot for the evaluation
plt.figure(figsize=(12, 6))
plt.plot(evaluation_steps, rewards_model1, label="Model 1 Rewards", marker='o')
plt.plot(evaluation_steps, rewards_model2, label="Model 2 Rewards", marker='x')
plt.axhline(100, color='green', linestyle='--', label="Maximum Reward (100)")
plt.xlabel("Evaluation Step")
plt.ylabel("Rewards")
plt.title("Evaluation Performance of Models")
plt.legend()
plt.grid()
plt.show()
