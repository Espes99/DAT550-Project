from torch.utils.data import DataLoader
from utils.rnn_preprocessing import ArxivDataset

def get_dataloaders_from_splits(train_df, val_df, batch_size=32, pad_col="padded", label_col="label_idx", shuffle=True):

    train_data = list(zip(train_df[pad_col], train_df[label_col]))
    val_data = list(zip(val_df[pad_col], val_df[label_col]))

    train_ds = ArxivDataset(train_data)
    val_ds = ArxivDataset(val_data)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    return train_loader, val_loader