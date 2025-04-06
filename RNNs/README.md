# Recurrent Neural Networks

## Folder structure
```bash
RNNs/
├── data/
│   └── arxiv10.csv.gz           # The dataset
├── notebooks/
│   ├── data_inspection.ipynb    # Quick way to inspect the dataset
│   └── rnn_experiments.ipynb    # Quick prototyping
├── models/
│   └── rnn_model.py             # RNN model architecture
├── utils/
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
