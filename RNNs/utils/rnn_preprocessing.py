import re
import torch
import pandas as pd
import pickle
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

class RNN_Preprocessor:
    """
    Preprocesses text data for RNN-based classification.
    """
    def __init__(self, dataFrame: pd.DataFrame = None, config: dict = None):
        config = config or {}
        # Hyper-parameters:
        self.MAX_LEN = config.get("max_len", 350) # Max length of an input, required for RNN processing

        # -----
        self.df = dataFrame if dataFrame is not None else pd.DataFrame()
        self.tokenizer = get_tokenizer("basic_english")
        self.label_encoder = LabelEncoder()


    def preprocess(self):
        self.df["clean_text"] = self.df["abstract"].apply(lambda x: self.clean_text(x))

        self.vocab = build_vocab_from_iterator(self.yield_tokens(self.df["clean_text"]), specials=["<unk>", "<pad>"])
        self.vocab.set_default_index(self.vocab["<unk>"])

        self.df["encoded"] = self.df["clean_text"].apply(self.encode_text)

        self.df["padded"] = self.df["encoded"].apply(lambda x: self.pad_sequence_fixed_length(x))
        
        self.df["label_idx"] = self.label_encoder.fit_transform(self.df["label"])

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'\$.*?\$', ' <MATH> ', text)
        text = re.sub(r'http\S+|www\.\S+', ' <URL> ', text)
        text = re.sub(r'\d+', '<NUM>', text)
        text = re.sub(r'(?<!<)[^\w\s>](?!>)', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def yield_tokens(self, data_iter):
        for text in data_iter:
            yield self.tokenizer(text)

    def encode_text(self, text):
        return [self.vocab[token] for token in self.tokenizer(text)]
    
    def pad_sequence_fixed_length(self, seq):
        seq = seq[:self.MAX_LEN]
        return torch.tensor(seq + [self.vocab["<pad>"]] * (self.MAX_LEN - len(seq)))
    
    def decode_tensor(self, tensor):
        """Decodes a tensor of token indices back into words (excluding <pad>)."""
        inverse_vocab = {idx: word for word, idx in self.vocab.get_stoi().items()}
        return ' '.join(inverse_vocab.get(idx.item(), "<unk>") for idx in tensor if idx.item() != self.vocab["<pad>"])

    
    # Save / Load

    def save(self, path_prefix: str):
        torch.save(self.vocab, f"{path_prefix}_vocab.pt")
        with open(f"{path_prefix}_label_encoder.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)
        self.df.to_pickle(f"{path_prefix}_df.pkl")

    def load(self, path_prefix: str):
        self.vocab = torch.load(f"{path_prefix}_vocab.pt")
        with open(f"{path_prefix}_label_encoder.pkl", "rb") as f:
            self.label_encoder = pickle.load(f)
        self.df = pd.read_pickle(f"{path_prefix}_df.pkl")

    # Getters / Setters
    def get_dataset(self):
        return list(zip(self.df["padded"], self.df["label_idx"]))
    
    def get_vocab(self):
        return self.vocab
    
    def get_label_encoder(self):
        return self.label_encoder
    
    def set_dataframe(self, df: pd.DataFrame):
        self.df = df
    
class ArxivDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]