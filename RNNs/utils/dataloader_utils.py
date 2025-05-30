from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from utils.rnn_preprocessing import ArxivDataset

def get_dataloaders_from_splits(train_df, val_df, batch_size=32, pad_col="padded", label_col="label_idx", shuffle=True):

    train_data = list(zip(train_df[pad_col], train_df[label_col]))
    val_data = list(zip(val_df[pad_col], val_df[label_col]))

    train_ds = ArxivDataset(train_data)
    val_ds = ArxivDataset(val_data)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    return train_loader, val_loader

def get_individual_dataloader(dataframe, batch_size=32, pad_col="padded", label_col="label_idx", shuffle=True):
    dataframe_data = list(zip(dataframe[pad_col], dataframe[label_col]))

    dataframe_ds = ArxivDataset(dataframe_data)

    dataframe_loader = DataLoader(dataframe_ds, batch_size=batch_size, shuffle=shuffle)

    return dataframe_loader

def split_func(dataframe):
    df1, df2 = train_test_split(dataframe, 0.8, 0.2, 42, stratify=dataframe["label"])
    return df1, df2