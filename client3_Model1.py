import flwr as fl
import sys
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from gym_examples.envs.slice_creation_env5_training2405 import SliceCreationEnv5
from gymnasium.wrappers import TimeLimit
from DQN_client_fw import DQNClient
from flwr.common import FitRes, Parameters, Status, Code, ndarrays_to_parameters

# Initialize environment and model
env = SliceCreationEnv5()
env = TimeLimit(env, max_episode_steps=99)
policy_kwargs = dict(net_arch=[32])

model = DQN("MlpPolicy", env, 
        buffer_size=int(1.5e5),
        learning_rate=1e-2,
        learning_starts=25000,
        exploration_fraction=0.25,
        exploration_final_eps=0,
        train_freq=4,
        gradient_steps=1,
        batch_size=64,
        gamma=0.99,
        tau=1.0,
        target_update_interval=500,
        verbose=0,
        policy_kwargs=policy_kwargs,
        device='cpu')

# Set client ID based on command-line argument to differentiate clients
client_id = 3 # Pass client ID as a command-line argument

class CustomDQNClient(DQNClient):
    def fit(self, fit_ins):
        config = fit_ins.config
        include_client_3 = config.get("include_client_3", True)
        round_num = config.get("round", 1)
        status = Status(code=Code.OK, message="Success")

        # Skip training if Client 3 until the last round
        if client_id == 3 and not include_client_3:
            print(f"Client {client_id} skipping round {round_num}")
            # Convert parameters to the correct format
            parameters = self.get_parameters()
            return FitRes(status, parameters.parameters, 0, {})

        # Train normally otherwise
        print(f"Client {client_id} training in round {round_num}")

        # Ensure we return a FitRes object with properly formatted parameters
        return super().fit(fit_ins)

# Initialize and start the client
client = CustomDQNClient(model, env, cid=client_id)
fl.client.start_client(server_address="localhost:8080", client=client)