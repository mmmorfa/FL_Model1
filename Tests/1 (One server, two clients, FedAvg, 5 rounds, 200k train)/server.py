import flwr as fl
from flwr.server.server import ServerConfig

# Define the number of rounds
NUM_ROUNDS = 5

# Configuration function to send to clients
def fit_config_fn(rnd: int) -> dict:
    # Set the save_model flag to True only on the last round
    save_model = rnd == NUM_ROUNDS
    return {"train_steps": 200000, "save_model": save_model, "round": rnd}

def model_1_strategy():
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=2,  # Minimum number of clients to participate in training
        min_evaluate_clients=2,  # Minimum number of clients to participate in evaluation
        min_available_clients=2,  # Minimum number of available clients required
        fraction_fit=1,  # Fraction of clients used in each round
        fraction_evaluate=0,  # Use none of the clients per round for evaluation
        on_fit_config_fn=fit_config_fn,
        #on_fit_config_fn=lambda round_num: {"train_steps": 500},  # Sending config to clients
        on_evaluate_config_fn=lambda round_num: {"eval_episodes": 5},  # Sending config to clients
    )
    return strategy

fl.server.start_server(
    server_address="localhost:8080",
    strategy=model_1_strategy(),
    config=ServerConfig(num_rounds=NUM_ROUNDS)
)