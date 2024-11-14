from stable_baselines3 import DQN

from stable_baselines3.common.logger import configure

from gym_examples.envs.slice_creation_env5_training2405 import SliceCreationEnv5

from gymnasium.wrappers import TimeLimit

from DQN_client_fw import DQNClient
import flwr as fl

env = SliceCreationEnv5()
env = TimeLimit(env, max_episode_steps=99)

#log_path = "/home/mario/Documents/Joint RAN-MEC Slicing & RA O-RAN/logs"
#new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])

policy_kwargs = dict(net_arch=[32])

model = DQN("MlpPolicy",env, 
        buffer_size=int(1.5e5),  # Replay buffer size
        learning_rate=1e-2,     # Learning rate
        learning_starts=25000,  # Number of steps before learning starts
        exploration_fraction=0.25,  # Fraction of total timesteps for exploration
        exploration_final_eps=0,  # Final exploration probability after exploration_fraction * total_timesteps
        train_freq=4,           # Update the model every `train_freq` steps
        gradient_steps=1,       # Number of gradient steps to take after each batch of data
        batch_size=64,          # Number of samples in each batch
        gamma=0.99,             # Discount factor
        tau=1.0,                # Target network update rate
        target_update_interval=500,  # Interval (in timesteps) at which the target network is updated
        verbose=0,              # Verbosity level
        policy_kwargs=policy_kwargs,
        device='cpu')              

#model.set_logger(new_logger)
#model.learn(total_timesteps=500000, log_interval=1)
#model.save("/home/mario/Documents/DQN_Models/Joint/gym-examples5/dqn_slices1_2305_arch32_500k_025epsilon_rb150k_tau1_learningstart50k_batch64_target500_learnrate-2")

client = DQNClient(model, env, cid=2)
fl.client.start_client(server_address="localhost:8080", client=client)