from pathlib import Path
import yaml
import os

# Base config that is shared among all runs
base_config = {
    "test_flight": False,
    "preprocess": {
        "max_len": 350
    },
    "training": {
        "epochs": 10,
        "lr": 0.001,
        "batch_size": 64,
        "early_stopping": True,
        "early_stopping_patience": 10,
        "scheduler": {
            "name": "ReduceLROnPlateau",
            "mode": "max",
            "factor": 0.5,
            "patience": 2,
            "verbose": True
        }
    },
    "model": {
        "return_attn_weights": True,
        "hidden_dim": 128
    },
    "batch_size": 64,
    "use_tensorboard": False,
    "evaluate_on_test": False,
    "run_type": "training"
}

# List of variations from the table
runs = [
    ("lstm", False, "none", "glove", 50, True, 1, 0.3, 0.001, "lstmNoneGlove50"),
    ("gru", False, "none", "glove", 50, True, 1, 0.3, 0.001, "gruNoneGlove50"),
    ("lstm", True, "none", "glove", 50, True, 1, 0.3, 0.001, "lstmBiNoneGlove50"),
    ("lstm", True, "custom", "glove", 50, True, 2, 0.3, 0.001, "lstmBiCustomGlove50_2L"),
    ("gru", True, "custom_mlp", "fasttext", 300, True, 1, 0.3, 0.001, "gruBiMLPFast300"),
    ("lstm", True, "custom_dot", "glove", 100, False, 1, 0.3, 0.001, "lstmBiDotGlove100Train"),
    ("lstm", True, "mha", "fasttext", 300, True, 2, 0.3, 0.001, "lstmBiMHAFast300_2L"),
    ("gru", False, "none", "random", 50, True, 1, 0.3, 0.001, "gruNoneRand50"),
    ("lstm", False, "none", "glove", 50, True, 1, 0.1, 0.001, "lstmNoneGlove50Drop1"),
    ("lstm", False, "none", "glove", 50, True, 1, 0.3, 0.01, "lstmNoneGlove50LR01"),
    ("lstm", True, "custom", "glove", 50, True, 1, 0.1, 0.001, "lstmBiCustomGlove50Drop1"),
    ("gru", True, "custom_mlp", "fasttext", 300, True, 1, 0.3, 0.01, "gruBiMLPFast300LR01")
]

# Output folder
output_dir = "configs/training"
#output_dir.mkdir(parents=True, exist_ok=True)

# Generate YAML files
for rnn, bidir, attn, emb_type, emb_dim, freeze, layers, dropout, lr, name in runs:
    config = base_config.copy()
    config["training"] = config["training"].copy()
    config["training"]["lr"] = lr
    config["model"] = config["model"].copy()
    config["model"].update({
        "rnn_type": rnn,
        "bidirectional": bidir,
        "attention_layer": attn,
        "num_layers": layers,
        "dropout": dropout
    })
    config["embedding"] = {
        "embedding_type": emb_type,
        "embedding_dim": emb_dim,
        "freeze": freeze
    }
    config["output_dir"] = f"outputs/training/{name}"
    config["log_dir"] = f"logs/training/{name}"
    config["test_type"] = name

    with open(os.path.join(output_dir, f"{name}.yaml"), "w") as f:
        yaml.dump(config, f)

str(output_dir)
