# Recurrent Neural Networks

## Folder structure
```bash
RNNs/
├── configs/
│   ├── rnn_standard_config.yaml     # Standard config file for training
│   └── rnn_test_flight.yaml         # Test config for experimental runs
├── logs/                            # Logs from all runs
├── models/
│   ├── __init__.py                  # Model package initializer
│   └── rnn_model.py                 # RNN model architecture
├── notebooks/
│   ├── data_inspection.ipynb        # Quick way to inspect the dataset
│   ├── Report_notes.ipynb           # Notes for the final report
│   └── rnn_prototyping.ipynb        # Prototyping and experimentation
├── outputs/                         # Folder for best models
├── test/
│   ├── __init__.py                  # Initialize test module
│   └── test_pipeline.py             # Unit tests for the pipeline
├── utils/
│   ├── __init__.py                  # Init for utils module
│   ├── dataloader_utils.py          # Dataset and DataLoader handling
│   ├── embedding_loader.py          # Embedding layer integration
│   └── rnn_preprocessing.py         # Tokenization, padding, etc.
├── __init__.py                      # Init for utils module
├── config.yaml                      # Global config file (possibly duplicate of configs/)
├── evaluate_rnn.py                  # Evaluation script
├── README.md                        # Project overview and instructions
├── run_training.py                  # Entry point for training the rnn
└── train_rnn.py                     # Training loop
```

## Execution
**All files contained in the RNNs folder need to be ran with RNNs as the origin folder**

## Testing procedure
Stand in the RNNs folder and run:
```bash
python -m pytest -v test/test_pipeline.py
```

TODO: TBC
