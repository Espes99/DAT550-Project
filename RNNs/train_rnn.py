from torch.utils.data import DataLoader
from utils.rnn_preprocessing import RNN_Preprocesser, ArxivDataset
import pandas as pd

# BOILER PLATE FOR FUNCTIONALITY NOT FINAL

df = pd.DataFrame
preprocesser = RNN_Preprocesser(df)
preprocesser.preprocess()

dataset = ArxivDataset(preprocesser.get_dataset())
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)