import torch
from torch.utils.data import DataLoader
from utils.rnn_preprocessing import RNN_Preprocesser, ArxivDataset
import pandas as pd
from utils.dataloader_utils import get_dataloaders_from_splits


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# BOILER PLATE FOR FUNCTIONALITY NOT FINAL

df = pd.DataFrame
preprocesser = RNN_Preprocesser(df)
preprocesser.preprocess()

dataset = ArxivDataset(preprocesser.get_dataset())
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# EXAMPLE OF dataloader_utils.py
train_df, val_df = pd.DataFrame
train_loader, val_loader = get_dataloaders_from_splits(train_df, val_df)