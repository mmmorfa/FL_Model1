import csv
import matplotlib.pyplot as plt

def read_csv(file_path):
    columns = {}  # Dictionary to store lists for each column
    
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        
        headers = next(reader, None)
        
        if headers:
            # Initialize lists for each column
            for header in headers:
                columns[header] = []
            
            # Read and store data in respective columns
            for row in reader:
                for header, value in zip(headers, row):
                    columns[header].append(value)
    
    return columns

#file_path1 = '/home/mario/Documents/DQN_Models/Model 1/gym-examples2/logs/progress_dqn_slices2(Arch:16; learn:1e-3; starts:250k; fraction:0_5; train: 1.5M).csv'


#file_path1 = '/home/mario/Documents/DQN_Models/Joint/gym-examples5/logs/progress.csv'
#file_path1 = '/home/mario/Documents/DQN_Models/Joint/gym-examples5/logs/progress(500k_04epsilon_rb150k_tau1_learningstart50k_batch64_target500).csv'

file_path1 = '/home/mario/Documents/DQN_Models/Joint/gym-examples5/logs/progress.csv'

#file_path2 = '/home/mario/Documents/DQN_Models/Model 1/gym-examples/logs/progress(1e-3).csv'
#file_path3 = '/home/mario/Documents/DQN_Models/Model 1/gym-examples/logs/progress(1e-5).csv'

data1 = read_csv(file_path1)
ep_rew_mean1 = [float(i) for i in data1['rollout/ep_rew_mean']]
ep_len_mean1 = [float(i) for i in data1['rollout/ep_len_mean']]
exploration_rate1 = [float(i) for i in data1['rollout/exploration_rate']]
episodes1 = [float(i) for i in data1['time/episodes']]
fps1 = [float(i) for i in data1['time/fps']]
time_elapsed1 = [float(i) for i in data1['time/time_elapsed']]
total_timesteps1 = [float(i) for i in data1['time/total_timesteps']]
learning_rate1 = [float(i) for i in data1['train/learning_rate'] if i!= '']
loss1 = [float(i) for i in data1['train/loss'] if i!= '']
n_updates1 = [float(i) for i in data1['train/n_updates'] if i!= '']

'''
data2 = read_csv(file_path2)
ep_rew_mean2 = [float(i) for i in data2['rollout/ep_rew_mean']]
ep_len_mean2 = [float(i) for i in data2['rollout/ep_len_mean']]
exploration_rate2 = [float(i) for i in data2['rollout/exploration_rate']]
episodes2 = [float(i) for i in data2['time/episodes']]
fps2 = [float(i) for i in data2['time/fps']]
time_elapsed2 = [float(i) for i in data2['time/time_elapsed']]
total_timesteps2 = [float(i) for i in data2['time/total_timesteps']]
learning_rate2 = [float(i) for i in data2['train/learning_rate'] if i!= '']
loss2 = [float(i) for i in data2['train/loss'] if i!= '']
n_updates2 = [float(i) for i in data2['train/n_updates'] if i!= '']

data3 = read_csv(file_path3)
ep_rew_mean3 = [float(i) for i in data3['rollout/ep_rew_mean']]
ep_len_mean3 = [float(i) for i in data3['rollout/ep_len_mean']]
exploration_rate3 = [float(i) for i in data3['rollout/exploration_rate']]
episodes3 = [float(i) for i in data3['time/episodes']]
fps3 = [float(i) for i in data3['time/fps']]
time_elapsed3 = [float(i) for i in data3['time/time_elapsed']]
total_timesteps3 = [float(i) for i in data3['time/total_timesteps']]
learning_rate3 = [float(i) for i in data3['train/learning_rate'] if i!= '']
loss3 = [float(i) for i in data3['train/loss'] if i!= '']
n_updates3 = [float(i) for i in data3['train/n_updates'] if i!= '']
'''

a = data1['train/loss'].index(str(loss1[0]))
#b = data2['train/loss'].index(str(loss2[0]))
#c = data3['train/loss'].index(str(loss3[0]))
#print (a, b, c)

plt.figure(1)
plt.rcParams.update({'font.size': 15})
plt.subplot(211)
plt.plot(total_timesteps1, ep_len_mean1, marker='.', linestyle='-', color='r', label='Reward')
plt.xlabel("Timesteps")
plt.ylabel("Reward")
plt.legend()
plt.subplot(212)
plt.plot(total_timesteps1, exploration_rate1, marker='.', linestyle='-', color='y', label='Exploration Rate')
#plt.plot(episodes1[a:], ep_rew_mean1[a:], marker='o', linestyle='-', color='g', label='Episode Mean Reward')
plt.xlabel("Timesteps")
plt.ylabel("Epsilon")
plt.legend()
plt.show()


