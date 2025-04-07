# Recurrent Neural Networks

## Folder structure
```bash
RNNs/
├── logs/
│   └── *.csv                    # Logs in a csv format
├── models/
│   └── rnn_model.py             # RNN model architecture
├── notebooks/
│   ├── data_inspection.ipynb    # Quick way to inspect the dataset
│   └── rnn_experiments.ipynb    # Quick prototyping
├── test/
│   ├── __init__.py              # Initializing the folder as a package
│   └── test_pipeline.py         # Test for utils package
├── utils/
│   ├── __init__.py              # Initializing the folder as a package
│   ├── dataloader_utils.py      # Helper function for loading the datasets
│   ├── embedding_loader.py      # Class for the embedding layer
│   └── rnn_preprocessing.py     # RNN specific preprocessing tokenization, padding, etc.
├── train_rnn.py                 # Training loop
├── evaluate_rnn.py              # Evaluation script
├── config.yaml                  # Parameters like batch size, learning rate
└── README.md
```

## Execution
**All files contained in the RNNs folder need to be ran with RNNs as the origin folder**

## Testing procedure
Stand in the RNNs folder and run:
```bash
python -m pytest -v test/test_pipeline.py
```

TODO: TBC
