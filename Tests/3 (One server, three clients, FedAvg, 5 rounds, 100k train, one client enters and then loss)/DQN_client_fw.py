import flwr as fl
import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, Status, GetParametersRes, Code, FitRes, FitIns, EvaluateIns, EvaluateRes
import torch as th
from stable_baselines3.common.logger import configure
import os
import csv
import sys

class DQNClient(fl.client.Client):
    def __init__(self, model, environment, cid):
        self.model = model
        self.environment = environment
        self.cid = cid      # Unique identifier for each client

        # Step 2: Configure a unique logger for each client
        log_dir = f"logs/client_{self.cid}"
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure the logger to output logs in the desired formats
        self.logger = configure(log_dir, ["csv", "json", "tensorboard"])

        # Attach the logger to the DQN model
        self.model.set_logger(self.logger)

    def get_parameters(self, config=None):
        # Fetch parameters from the model
        param_dict = self.model.get_parameters()
        
        # Initialize list to hold numpy-converted parameters
        param_list = []
        
        # Flatten and convert each tensor to a NumPy array
        # In most federated learning setups, only the learnable parameters (the tensors associated with weights and biases of neural network layers) 
        # need to be shared. Other items in the model that are not tensors (e.g., metadata or non-trainable values) can be ignored for Flower.
        for state_dict in param_dict.values():
            for tensor in state_dict.values():
                if isinstance(tensor, th.Tensor):
                    param_list.append(tensor.detach().cpu().numpy())
        
        # Convert list of numpy arrays to Flower's Parameters object
        parameters = ndarrays_to_parameters(param_list)
        
        # Create a Status object indicating success
        status = Status(code=Code.OK, message="Success")
        
        # Return a GetParametersRes object with Parameters and Status
        return GetParametersRes (status=status, parameters=parameters)
    
    
    def set_parameters(self, parameters):
        # Convert Flower's Parameters object back into a list of numpy arrays
        param_list = parameters_to_ndarrays(parameters)
        
        # Step 2: Get the parameter structure of the model to update it correctly
        param_dict = self.model.get_parameters()
            
        # Initialize index to track the position in the param_list
        idx = 0
        
        # Reconstruct the model's parameter dictionary
        for state_dict in param_dict.values():
            for key in state_dict:
                if isinstance(state_dict[key], th.Tensor):
                    # Update only learnable parameters from param_list
                    state_dict[key] = th.tensor(param_list[idx], device=self.model.device)
                    idx += 1
                else:
                    # Initialize or retain non-learnable parameters
                    state_dict[key] = state_dict[key]
        
        # Set the updated parameters back into the model
        self.model.set_parameters(param_dict)

    def fit(self, fit_ins: FitIns) -> FitRes:
        # Extract parameters and config from FitIns
        parameters = fit_ins.parameters
        config = fit_ins.config
        
        # Set the model's parameters received from the server
        self.set_parameters(parameters)
        
        # Retrieve the number of training steps from the config and the flag to save the model after training
        train_steps = config.get("train_steps", 500)  # Default to 1000 steps if not specified
        save_model = config.get("save_model", False)
        round_num = config.get("round", 0)  # Retrieve the round number from the config
        
        # Train the model locally
        self.model.learn(total_timesteps=train_steps, log_interval=1)

        # Get the updated parameters after training
        updated_parameters = self.get_parameters().parameters
        
        # Save the model if instructed by the server
        if save_model:
            model_path = f"Trained_Models/client_{self.cid}_model_round_{round_num}.zip"
            self.model.save(model_path)
            print(f"Model saved for client {self.cid} at {model_path}")

        # Number of examples can be the size of local data; assume it as 1 for illustration
        num_examples = 1
        
        # Optionally, add training metrics if available (e.g., loss)
        metrics = {"train_steps": train_steps}

        status = Status(code=Code.OK, message="Success")
        
        # Return FitRes containing updated parameters, number of examples, and metrics
        return FitRes(
            status=status,
            parameters=updated_parameters,
            num_examples=num_examples,
            metrics=metrics
        )

    """def evaluate(self, eval_ins: EvaluateIns) -> EvaluateRes:
        # Set the model parameters to the received parameters for evaluation
        self.set_parameters(eval_ins.parameters)
        
        # Initialize variables for rewards and TD errors
        rewards = []
        td_errors = []
        
        # Perform evaluation for the specified number of episodes
        for _ in range(eval_ins.config.get("eval_episodes", 1)):  # Default to 1 episode if not specified
            obs = self.environment.reset()
            done, reward_sum = False, 0
            while not done:
                # Get the action and Q-value prediction from the model
                action, q_values = self.model.predict(obs, deterministic=True)
                
                # Take a step in the environment
                next_obs, reward, done, info = self.environment.step(action)
                reward_sum += reward
                
                # Calculate TD error: TD_error = reward + gamma * max(Q(next_obs)) - Q(obs, action)
                next_q_values = self.model.predict(next_obs, deterministic=True)[1]
                td_error = reward + self.model.gamma * max(next_q_values) - q_values[action]
                td_errors.append(td_error.item())  # Store TD error as a scalar
                
                # Move to the next observation
                obs = next_obs
            
            rewards.append(reward_sum)
        
        # Calculate the average reward and TD error
        average_reward = float(sum(rewards) / len(rewards))
        average_td_error = float(sum(td_errors) / len(td_errors)) if td_errors else 0.0
        
        # Create a Status object indicating success
        status = Status(code=Code.OK, message="Evaluation successful")
        
        # Number of examples used in evaluation (set to 1 as placeholder)
        num_examples = 1
        
        # Metrics can include both average reward and average TD error
        metrics = {"average_reward": average_reward, "average_td_error": average_td_error}
        
        # Return EvaluateRes with TD error as the loss
        return EvaluateRes(
            status=status,
            loss=average_td_error,  # Use TD error as "loss"
            num_examples=num_examples,
            metrics=metrics
        )"""

# test
    # Helper functions for resource utilization calculations
    def calculate_utilization_mec(self, parameter, current, total, utilization_list):
        utilization = ((total - current) / total) * 100
        utilization_list.append(utilization)

    def calculate_utilization_ran(self, bwp, current, utilization_list):
        indices = np.where(current == 0)
        available_symbols = len(indices[0])
        utilization = ((current.size - available_symbols) / current.size) * 100
        utilization_list.append(utilization)

    def save_evaluation_to_csv(self, round_number, rewards, mec_cpu_utilization, mec_ram_utilization, mec_storage_utilization, mec_bw_utilization, ran_bwp1_utilization, ran_bwp2_utilization):
            # Set up the folder path for logs
            log_dir = f"logs/client_{self.cid}"
            os.makedirs(log_dir, exist_ok=True)
            file_path = os.path.join(log_dir, f"evaluation_round_{round_number}.csv")

            # Prepare data for CSV
            data = {
                "rewards": rewards,
                "mec_cpu_utilization": mec_cpu_utilization,
                "mec_ram_utilization": mec_ram_utilization,
                "mec_storage_utilization": mec_storage_utilization,
                "mec_bw_utilization": mec_bw_utilization,
                "ran_bwp1_utilization": ran_bwp1_utilization,
                "ran_bwp2_utilization": ran_bwp2_utilization
            }

            # Determine the maximum length of lists to pad shorter lists
            max_len = max(len(lst) for lst in data.values())

            # Pad lists to the same length
            padded_data = {key: lst + [None] * (max_len - len(lst)) for key, lst in data.items()}

            # Write data to CSV
            with open(file_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(padded_data.keys())  # Write header
                writer.writerows(zip(*padded_data.values()))  # Write data rows

    def calculate_bandwidth(parameters) -> int:
        """Calculate the bandwidth of serialized model parameters in bytes."""
        serialized_params = parameters.SerializeToString()
        return len(serialized_params)  # Return size in bytes

    def evaluate(self, eval_ins: EvaluateIns) -> EvaluateRes:
            # Set the model parameters to the received parameters for evaluation
            self.set_parameters(eval_ins.parameters)

            # Counter to limit the evaluation period
            counter_evaluation = 0

            # Initialize lists to store rewards and utilization metrics
            rewards = []
            mec_cpu_utilization = []
            mec_ram_utilization = []
            mec_storage_utilization = []
            mec_bw_utilization = []
            ran_bwp1_utilization = []
            ran_bwp2_utilization = []

            # Perform evaluation for the specified number of episodes
            for _ in range(eval_ins.config.get("eval_episodes", 1)):
                obs, _ = self.environment.reset()
                terminated, reward_sum = False, 0
                while counter_evaluation < 100:
                    # Predict action using the model
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.environment.step(action)
                    reward_sum += reward
                    counter_evaluation += 1

                    if terminated or truncated:
                        obs, _ = self.environment.reset()

                    # Calculate MEC resource utilization
                    self.calculate_utilization_mec('cpu', self.environment.unwrapped.resources_1['MEC_CPU'], 30, mec_cpu_utilization)
                    self.calculate_utilization_mec('ram', self.environment.unwrapped.resources_1['MEC_RAM'], 128, mec_ram_utilization)
                    self.calculate_utilization_mec('storage', self.environment.unwrapped.resources_1['MEC_STORAGE'], 1000, mec_storage_utilization)
                    self.calculate_utilization_mec('bw', self.environment.unwrapped.resources_1['MEC_BW'], 300, mec_bw_utilization)

                    # Calculate RAN utilization
                    self.calculate_utilization_ran('bwp1', self.environment.unwrapped.PRB_map1, ran_bwp1_utilization)
                    self.calculate_utilization_ran('bwp2', self.environment.unwrapped.PRB_map2, ran_bwp2_utilization)

                # Store the total reward for this episode
                rewards.append(reward_sum)

            # Save lists to CSV file after evaluation round
            round_number = eval_ins.config.get("round", 0)
            self.save_evaluation_to_csv(round_number, rewards, mec_cpu_utilization, mec_ram_utilization, mec_storage_utilization, mec_bw_utilization, ran_bwp1_utilization, ran_bwp2_utilization)

            # Calculate average reward
            average_reward = float(sum(rewards) / len(rewards))

            # Aggregate resource utilization statistics for this evaluation
            avg_mec_cpu_utilization = np.mean(mec_cpu_utilization) if mec_cpu_utilization else 0.0
            avg_mec_ram_utilization = np.mean(mec_ram_utilization) if mec_ram_utilization else 0.0
            avg_mec_storage_utilization = np.mean(mec_storage_utilization) if mec_storage_utilization else 0.0
            avg_mec_bw_utilization = np.mean(mec_bw_utilization) if mec_bw_utilization else 0.0
            avg_ran_bwp1_utilization = np.mean(ran_bwp1_utilization) if ran_bwp1_utilization else 0.0
            avg_ran_bwp2_utilization = np.mean(ran_bwp2_utilization) if ran_bwp2_utilization else 0.0

            # Create a Status object indicating success
            status = Status(code=Code.OK, message="Evaluation successful")

            # Number of examples used in evaluation
            num_examples = 1  # Placeholder, as we're evaluating on episodes rather than a dataset

            # Metrics for resource utilization
            metrics = {
                "average_reward": average_reward,
                "avg_mec_cpu_utilization": avg_mec_cpu_utilization,
                "avg_mec_ram_utilization": avg_mec_ram_utilization,
                "avg_mec_storage_utilization": avg_mec_storage_utilization,
                "avg_mec_bw_utilization": avg_mec_bw_utilization,
                "avg_ran_bwp1_utilization": avg_ran_bwp1_utilization,
                "avg_ran_bwp2_utilization": avg_ran_bwp2_utilization,
            }

            # Return EvaluateRes with calculated utilization as metrics
            return EvaluateRes(
                status=status,
                loss=-average_reward,  # Negative reward as a form of loss
                num_examples=num_examples,
                metrics=metrics
            )
    