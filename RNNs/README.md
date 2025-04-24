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

## General information
This model is configured to try running the training and evaluation scripts on the "CUDA", "MPS" (Apple silicon macs), "CPU" in this order defaulting to cpu processing.
If you want to run any processing on cuda cores you need install the cuda version of pytorch. Because we use "torchtext" in this project we are limited to the version of torchtext that is compatible with torch, at the time of development the newest version of torchtext is 0.18.0, and the latest compatible torch version with cuda core support is torch 2.3.0+cu121.

It's recommended to first install torchtext, uninstall torch, and then install a version of torch with cuda. This is done because torchtext will demand installing its own version of torch, however this version will not support cuda.

## Execution
**All files contained in the RNNs folder need to be ran with RNNs as the origin folder**

## Testing procedure
Stand in the RNNs folder and run:
```bash
python -m pytest -v test/test_pipeline.py
```

## Training a model
To train a model, configure a training config and run the following command:
```bash
python run_training.py --config configs/training/*.yaml # *.yaml is not a valid file to run, its to indicate that you can run any yaml file in this directory
# This command can be ran as a standalone without flags, that will default to the config "rnn_test_flight.yaml"
```

The available flags are:
```bash
--config "Path to config file"
--config_dir "Path to directory for a set of config files to load multiple configs, example: config/training/set1"
```

## Evaluating a model
To evaluate a model configure a config script, certain values must remain the same between training and evaluation, these configurations will be loaded automatically by the evaluation procedure. To run an evaluation execute the following command:
```bash
python evaluate_rnn.py --config configs/testing/eval_test.yaml
```
The available flags are:
```bash
--config "Path to config file"
--config_dir "Path to directory for a set of config files to load multiple configs"
```

## Analysis procedure
### General analysis
Run the following command with a list of model directories:
```bash
python analyze_metrics.py --runs outputs/training/testGruBiCustomGloveFSched outputs/training/testLstmNonbiCustomDotGloveFSched outputs/training/testLstmNonbiNoneGloveFSched --mode both
```
The available flags are:
```bash
--runs "list all model directories with a separating space"
--mode options are ["train", "test", "both"] # recommended to run "both" for best stability
```
Commands used in this project:
```bash
# Final small test run just to test 256 efficiency with unfrozen (train/eval commands)
python run_training.py --config configs/training/baselineHD256Unfrozen.yaml
python evaluate_rnn.py --config configs/testing/baselineHD256Unfrozen.yaml

# Config set 1
python analyze_metrics.py --runs outputs/training/gruBiMLPFast300/gruBiMLPFast300 outputs/training/gruBiMLPFast300LR01/gruBiMLPFast300LR01 outputs/training/gruNoneGlove50/gruNoneGlove50 outputs/training/gruNoneRand50/gruNoneRand50 --mode both

python analyze_metrics.py --runs outputs/training/lstmBiCustomGlove50_2L/lstmBiCustomGlove50_2L outputs/training/lstmBiCustomGlove50Drop1/lstmBiCustomGlove50Drop1 --mode both

# Config set 2
python analyze_metrics.py --runs outputs/training/lstmBiDotGlove100Train/lstmBiDotGlove100Train outputs/training/lstmBiMHAFast300_2L/lstmBiMHAFast300_2L outputs/training/lstmBiNoneGlove50/lstmBiNoneGlove50 outputs/training/lstmNoneGlove50/lstmNoneGlove50 --mode both

python analyze_metrics.py --runs outputs/training/lstmNoneGlove50Drop1/lstmNoneGlove50Drop1 outputs/training/lstmNoneGlove50LR01/lstmNoneGlove50LR01 --mode both

# Combination of the second in set 1 and set 2
python analyze_metrics.py --runs outputs/training/lstmBiCustomGlove50_2L/lstmBiCustomGlove50_2L outputs/training/lstmBiCustomGlove50Drop1/lstmBiCustomGlove50Drop1 outputs/training/lstmNoneGlove50Drop1/lstmNoneGlove50Drop1 outputs/training/lstmNoneGlove50LR01/lstmNoneGlove50LR01 --mode both

# Final config set
python analyze_metrics.py --runs outputs/training/baseline/baseline outputs/training/baselineDrop02/baselineDrop02 outputs/training/baselineHD256/baselineHD256 outputs/training/baselineUnfrozen/baselineUnfrozen --mode both

# Small final test
python analyze_metrics.py --runs outputs/training/baseline/baseline outputs/training/baselineHD256/baselineHD256 outputs/training/baselineUnfrozen/baselineUnfrozen outputs/training/baselineHD256Unfrozen/baselineHD256Unfrozen --mode both

#Final graph setup for High avg low performing models
python analyze_metrics.py --runs outputs/training/gruBiMLPFast300/gruBiMLPFast300 outputs/training/lstmBiNoneGlove50/lstmBiNoneGlove50 outputs/training/lstmNoneGlove50/lstmNoneGlove50 outputs/training/gruNoneGlove50/gruNoneGlove50 --mode both

```

### Attention evolution analysis
**Small note this code needs to be run with both training and evaluation metrics stored in the output folder**\
**You can only analyze 4 models at a time**\
Run the following command:
```bash
python plot_animated_attention.py --trace_dir outputs/training/testGruBiCustomGloveFSched/attention_trace --out attention_evolution.gif
```
The available flags are:
```bash
--trace_dir "Directory containing the models attention trace"
--out "Output file remember to use format '.gif'"
```
TODO: TBC
