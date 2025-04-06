import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchtext")

from utils.rnn_preprocessing import RNN_Preprocesser
from utils.dataloader_utils import get_dataloaders_from_splits

def test_data_pipeline():
    df = pd.read_csv("../Data/arxiv_train.csv")

    preprocessor = RNN_Preprocesser(df)
    preprocessor.preprocess()

    train_df, val_df = train_test_split(
        preprocessor.df,
        test_size=0.1,
        stratify=preprocessor.df["label_idx"],
        random_state=42
    )

    train_loader, val_loader = get_dataloaders_from_splits(train_df, val_df, batch_size=32)

    for batch_x, batch_y in train_loader:
        assert batch_x.shape == (32, preprocessor.MAX_LEN), "Batch input shape mismatch"
        assert batch_y.shape == (32,), "Batch label shape mismatch"
        assert batch_x.dtype == torch.int64, "Unexpected tensor dtype for input"
        assert batch_y.dtype == torch.int64, "Unexpected tensor dtype for labels"
        break
