import flwr as fl
from flwr.server.server import ServerConfig
from flwr.server.client_manager import SimpleClientManager

# Define the number of rounds
NUM_ROUNDS = 5

# Configuration function to send to clients
def fit_config_fn(rnd: int) -> dict:
    save_model = rnd == NUM_ROUNDS  # Save model only in the last round
    include_client_3 = rnd in {1, 2, NUM_ROUNDS}  # Include client 3 in rounds 1, 2, and the last round
    return {"train_steps": 100, "save_model": save_model, "round": rnd, "include_client_3": include_client_3}

def model_1_strategy():
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=2,  # Minimum number of clients to participate in training
        min_evaluate_clients=2,  # Minimum number of clients to participate in evaluation
        min_available_clients=2,  # Minimum number of available clients required
        fraction_fit=1.0,  # All clients are available to select
        fraction_evaluate=1.0,  # Skip evaluation
        on_fit_config_fn=fit_config_fn,  # Custom config for each round
    )
    return strategy

fl.server.start_server(
    server_address="localhost:8080",
    strategy=model_1_strategy(),
    config=ServerConfig(num_rounds=NUM_ROUNDS)
)