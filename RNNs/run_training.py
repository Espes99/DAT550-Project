import yaml
import argparse
from train_rnn import RNNTrainer

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to YAML config", default="configs/rnn_test_flight.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    trainer = RNNTrainer(config)
    trainer.train()