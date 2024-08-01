import torch
import os
import logging
import datetime
from src_batch.instance_creator.InstanceGenerator import InstanceGenerator
from src_batch.model.Model import Model

timestamp: str = datetime.datetime.now().strftime("%m_%d_%Y_%H:%M:%S")

try:
    logging.config.fileConfig(
        "logging.ini",
        disable_existing_loggers=False,
        defaults={
            "logfilename": datetime.datetime.now().strftime(
                f"logs/g-routing_{timestamp}.log"
            )
        },
    )
except:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

logger = logging.getLogger(__name__)
logger.debug("Debug Mode is enabled")

# CUDA is used if available, otherwise CPU is used.
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Running on device: {device}")


def prepare_data(data_path: str, batch_size: int = 1) -> tuple:
    IG: InstanceGenerator = InstanceGenerator()
    data_loader = IG.get_dataloader(data_path, batch_size=batch_size)
    capacity: float = data_loader.dataset[0].capacity
    return data_loader, capacity


def load_model(
    model_path: str,
    device: torch.device,
    node_input_dim: int = 1,
    edge_input_dim: int = 1,
    hidden_dim: int = 128,
    dropout: float = 0.6,
    layers: int = 2,
    heads: int = 8,
    capacity: int = 1000,
    T: float = 1.0,
) -> Model:
    model: Model = Model(
        node_input_dim, edge_input_dim, hidden_dim, dropout, layers, heads, capacity, T
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def run_inference(
    model: Model,
    data_loader: any,
    n_steps: int,
    greedy: bool,
    T: float,
) -> None:
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            actions, tour_logp, depot_visits = model(batch, n_steps, greedy, T)
            print("Actions: ", actions)
            print("Tour Log Probabilities: ", tour_logp)
            print("Depot Visits: ", depot_visits)


def main():

    # TODO - Don't Hardcode paths
    # Define paths
    model_path: str = r"model_checkpoints\149\actor.pt"
    data_path: str = (
        r"D:\DAY2DAY\MESTRADO\Codes\GNN\GAT_VRP1\gat_vrp1\src_batch\instances\train\train_20.CSV"
    )
    # data_path = r'D:\DAY2DAY\MESTRADO\Codes\GNN\GAT_VRP1\gat_vrp1\src_batch\instances\test\test_20nodes_TSP_100.CSV'

    # Params
    node_input_dim: int = 1
    edge_input_dim: int = 1
    hidden_dim: int = 128
    dropout: float = 0.6
    layers: int = 2
    heads: int = 4
    n_steps: int = 1
    greedy: bool = True
    T: float = 2.5  # Temperature for softmax based on Kun et al. (2021)
    batch_size: int = 10

    # Prepare the data
    data_loader, capacity = prepare_data(data_path, batch_size)
    for batch in data_loader:
        print(batch)

    # Load the pre-trained model
    model: Model = load_model(
        model_path,
        device,
        node_input_dim,
        edge_input_dim,
        hidden_dim,
        dropout,
        layers,
        heads,
        capacity,
        T,
    )

    # Run inference
    run_inference(model, data_loader, n_steps, greedy, T)


if __name__ == "__main__":
    main()
    logger.info(timestamp)
