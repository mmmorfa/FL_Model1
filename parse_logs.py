import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def parse_and_plot_all_rounds(log_files, train_steps=200000):
    # Iterate over each client log file
    for log_file in log_files:
        client_id = log_file.split('/')[-2]  # Extract client ID from file path
        df = pd.read_csv(log_file)
        
        # Identify rounds by detecting where "time/total_timesteps" resets to 1
        rounds = []
        current_round = []
        
        for _, row in df.iterrows():
            if row['time/episodes'] == 1 and current_round:
                rounds.append(pd.DataFrame(current_round))
                current_round = []
            current_round.append(row)
        
        # Append the last round
        if current_round:
            rounds.append(pd.DataFrame(current_round))
        
        # Plot all rounds in a single figure for each metric
        plt.figure(figsize=(15, 10))
        plt.suptitle(f"Training Metrics for {client_id}", fontsize=16)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(rounds)))  # Color scheme for different rounds

        # Plot 1: Episode Reward Mean
        plt.subplot(2, 2, 1)
        for round_idx, round_df in enumerate(rounds, start=1):
            plt.plot(round_df['time/total_timesteps'], round_df['rollout/ep_rew_mean'], 
                     label=f'Round {round_idx}', color=colors[round_idx - 1])
        plt.title('Mean Episode Reward Over Time')
        plt.xlabel('Total Timesteps')
        plt.ylabel('Mean Reward')
        plt.grid()
        plt.legend()

        # Plot 2: Exploration Rate
        plt.subplot(2, 2, 2)
        for round_idx, round_df in enumerate(rounds, start=1):
            plt.plot(round_df['time/total_timesteps'], round_df['rollout/exploration_rate'], 
                     label=f'Round {round_idx}', color=colors[round_idx - 1])
        plt.title('Exploration Rate Over Time')
        plt.xlabel('Total Timesteps')
        plt.ylabel('Exploration Rate')
        plt.grid()
        plt.legend()

        # Plot 3: Training Loss
        plt.subplot(2, 2, 3)
        for round_idx, round_df in enumerate(rounds, start=1):
            plt.plot(round_df['train/n_updates'], round_df['train/loss'], 
                     label=f'Round {round_idx}', color=colors[round_idx - 1])
        plt.title('Training Loss Over Updates')
        plt.xlabel('Number of Updates')
        plt.ylabel('Loss')
        plt.grid()
        plt.legend()

        # Plot 4: Episode Length Mean
        plt.subplot(2, 2, 4)
        for round_idx, round_df in enumerate(rounds, start=1):
            plt.plot(round_df['time/total_timesteps'], round_df['rollout/ep_len_mean'], 
                     label=f'Round {round_idx}', color=colors[round_idx - 1])
        plt.title('Mean Episode Length Over Time')
        plt.xlabel('Total Timesteps')
        plt.ylabel('Mean Episode Length')
        plt.grid()
        plt.legend()

        # Adjust layout and show the plots for the current client
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Add space for the client title
        plt.show()

# Example usage with paths to multiple client log files
log_files = [
    'logs/client_1/progress.csv',
    'logs/client_2/progress.csv',
    'logs/client_3/progress.csv',
    # Add more clients as needed
]
parse_and_plot_all_rounds(log_files, train_steps=100000)
