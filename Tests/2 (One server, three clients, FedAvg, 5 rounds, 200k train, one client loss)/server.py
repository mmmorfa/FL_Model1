import flwr as fl
from flwr.server.server import ServerConfig
from flwr.server.client_manager import SimpleClientManager

# Define the number of rounds
NUM_ROUNDS = 5

# Configuration function to send to clients
def fit_config_fn(rnd: int) -> dict:
    save_model = rnd == NUM_ROUNDS  # Save model only in the last round
    include_client_3 = rnd == NUM_ROUNDS  # Include client 3 only in the last round
    return {"train_steps": 200000, "save_model": save_model, "round": rnd, "include_client_3": include_client_3}

def model_1_strategy():
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=2,  # Minimum number of clients to participate in training
        min_evaluate_clients=2,  # Minimum number of clients to participate in evaluation
        min_available_clients=3,  # Minimum number of available clients required
        fraction_fit=1.0,  # All clients are available to select
        fraction_evaluate=0,  # Skip evaluation
        on_fit_config_fn=fit_config_fn,  # Custom config for each round
    )
    return strategy

# Custom Flower server class to inject post-round update logic
class CustomServer(fl.server.Server):
    def on_round_end(self, round_number, results, failures):
        # After round 4, push the global model to Client 3
        if round_number == NUM_ROUNDS - 1:
            print("Updating Client 3 with the global model after round 4")
            client_3 = self.client_manager().get_client(3)
            self.strategy.configure_fit(client_3)


# Initialize the custom server with a client manager
client_manager = SimpleClientManager()
server = CustomServer(client_manager=client_manager, strategy=model_1_strategy())

# Start the custom server
fl.server.start_server(
    server_address="localhost:8080",
    strategy=model_1_strategy(),
    config=ServerConfig(num_rounds=NUM_ROUNDS),
    server=server
)
