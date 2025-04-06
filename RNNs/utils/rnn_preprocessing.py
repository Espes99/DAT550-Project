import re
import torch
import pandas as pd
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder

class RNN_Preprocesser:
    def __init__(self, dataFrame: pd.DataFrame, config: dict = None):
        config = config or {}
        # Hyper-parameters:
        self.MAX_LEN = config.get("max_len", 350) # Max length of an input, required for RNN processing

        # -----
        self.df = dataFrame
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
    
    # Getters
    def get_dataset(self):
        return list(zip(self.df["padded"], self.df["label_idx"]))
    
    def get_vocab(self):
        return self.vocab
    
    def get_label_encoder(self):
        return self.label_encoder
    
