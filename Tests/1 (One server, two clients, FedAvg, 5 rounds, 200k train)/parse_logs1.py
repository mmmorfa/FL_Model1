import pandas as pd
import matplotlib.pyplot as plt

def parse_and_plot_per_round(log_file, train_steps=200000):
    # Step 1: Read the CSV file into a pandas DataFrame
    df = pd.read_csv(log_file)
    
    # Step 2: Split DataFrame by round (when `time/total_timesteps` resets to 1)
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
    
    # Step 3: Plot each round's data
    for round_idx, round_df in enumerate(rounds, start=1):
        time_elapsed = round_df['time/time_elapsed']
        ep_rew_mean = round_df['rollout/ep_rew_mean']
        total_timesteps = round_df['time/total_timesteps']
        episodes = round_df['time/episodes']
        exploration_rate = round_df['rollout/exploration_rate']
        ep_len_mean = round_df['rollout/ep_len_mean']
        loss = round_df['train/loss']
        n_updates = round_df['train/n_updates']
        learning_rate = round_df['train/learning_rate']

        # Create plots for the current round
        plt.figure(figsize=(15, 10))
        plt.suptitle(f"Training Metrics - Round {round_idx}", fontsize=16)

        # Plot 1: Episode Reward Mean
        plt.subplot(2, 2, 1)
        plt.plot(total_timesteps, ep_rew_mean, label='Mean Reward')
        plt.title('Mean Episode Reward Over Time')
        plt.xlabel('Total Timesteps')
        plt.ylabel('Mean Reward')
        plt.grid()
        plt.legend()

        # Plot 2: Exploration Rate
        plt.subplot(2, 2, 2)
        plt.plot(total_timesteps, exploration_rate, label='Exploration Rate', color='orange')
        plt.title('Exploration Rate Over Time')
        plt.xlabel('Total Timesteps')
        plt.ylabel('Exploration Rate')
        plt.grid()
        plt.legend()

        # Plot 3: Training Loss
        plt.subplot(2, 2, 3)
        plt.plot(n_updates, loss, label='Loss', color='green')
        plt.title('Training Loss Over Updates')
        plt.xlabel('Number of Updates')
        plt.ylabel('Loss')
        plt.grid()
        plt.legend()

        # Plot 4: Episode Length Mean
        plt.subplot(2, 2, 4)
        plt.plot(total_timesteps, ep_len_mean, label='Mean Episode Length', color='red')
        plt.title('Mean Episode Length Over Time')
        plt.xlabel('Total Timesteps')
        plt.ylabel('Mean Episode Length')
        plt.grid()
        plt.legend()

        # Adjust layout and show the plot for the current round
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Add space for the round title
        plt.show()

# Example usage
parse_and_plot_per_round('logs/client_1/progress.csv', train_steps=200000)
