import flwr as fl
from flwr.server.server import ServerConfig
from flwr.server.client_manager import SimpleClientManager
from pathlib import Path

# Define the number of rounds
NUM_ROUNDS = 5

# Configuration function to send to clients
def fit_config_fn(rnd: int) -> dict:
    save_model = rnd == NUM_ROUNDS  # Save model only in the last round
    return {"train_steps": 100, "save_model": save_model, "round": rnd}

def model_1_strategy():
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=4,  # Minimum number of clients to participate in training
        min_evaluate_clients=4,  # Minimum number of clients to participate in evaluation
        min_available_clients=4,  # Minimum number of available clients required
        fraction_fit=1.0,  # All clients are available to select
        fraction_evaluate=1.0,  # Skip evaluation
        on_fit_config_fn=fit_config_fn,  # Custom config for each round
    )
    return strategy

fl.server.start_server(
    server_address="localhost:8080",
    strategy=model_1_strategy(),
    config=ServerConfig(num_rounds=NUM_ROUNDS),
    certificates=(
        open("Certificates/rootCA.pem", "rb").read(),
        open("Certificates/server.pem", "rb").read(),
        open("Certificates/server.key", "rb").read(),
    )
)