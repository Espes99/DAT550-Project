import yaml
import os
import argparse
from train_rnn import RNNTrainer

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to YAML config", default="configs/training/rnn_test_flight.yaml")
    parser.add_argument("--config_dir", type=str, help="Directory path for all config files to be run, this overrides --config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    config_dir = args.config_dir

    if args.config_dir:
        config_dir = args.config_dir
        config_files = sorted([
            os.path.join(config_dir, f)
            for f in os.listdir(config_dir)
            if f.endswith(".yaml")
        ])
        print(f"Total configs registered: {len(config_files)}")
        for c in config_files:
            print(f"Starting config: {c}")
            loaded_c = load_config(c)
            trainer = RNNTrainer(loaded_c)
            trainer.train()
    else:
        print(f"Starting single config: {args.config}")
        config = load_config(args.config)
        trainer = RNNTrainer(config)
        trainer.train()