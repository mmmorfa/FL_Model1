from stable_baselines3 import DQN

from stable_baselines3.common.logger import configure

from gym_examples.envs.slice_creation_traffic_pattern1 import SliceCreationEnv5

from gymnasium.wrappers import TimeLimit

from DQN_client_fw import DQNClient
import flwr as fl

env = SliceCreationEnv5()
env = TimeLimit(env, max_episode_steps=99)

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

client = DQNClient(model, env, cid=1)
fl.client.start_client(server_address="localhost:8080", 
                       client=client,
                       root_certificates=open("Certificates/rootCA.pem", "rb").read()
                       )