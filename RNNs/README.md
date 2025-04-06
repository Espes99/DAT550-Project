# Recurrent Neural Networks

## Folder structure
```bash
RNNs/
├── notebooks/
│   ├── data_inspection.ipynb    # Quick way to inspect the dataset
│   └── rnn_experiments.ipynb    # Quick prototyping
├── test/
│   ├── __init__.py              # Initializing the folder as a package
│   └── test_pipeline.py         # Test for the data pipeline
├── models/
│   └── rnn_model.py             # RNN model architecture
├── utils/
│   ├── __init__.py              # Initializing the folder as a package
│   ├── dataloader_utils.py      # Helper function for loading the datasets 
│   └── rnn_preprocessing.py     # RNN specific preprocessing tokenization, padding, etc.
├── train_rnn.py                 # Training loop
├── evaluate_rnn.py              # Evaluation script
├── config.yaml                  # Parameters like batch size, learning rate
└── README.md
```

## Testing data pipeline
Stand in the RNNs folder and run:
```bash
python -m pytest -v test/test_pipeline.py
```

TODO: TBC
