import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchtext")

from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from utils.rnn_preprocessing import RNN_Preprocessor
from utils.dataloader_utils import get_dataloaders_from_splits
from utils.embedding_loader import EmbeddingLoader

def test_data_pipeline():
    df = pd.read_csv("../Data/arxiv_train.csv")

    preprocessor = RNN_Preprocessor(df)
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

def test_embedding_loader_pipeline():
    config_random = {"embedding_type": "random", "embedding_dim": 100, "freeze": False}
    config_glove = {"embedding_type": "glove", "embedding_dim": 100, "freeze": True}

    corpus = ["This is a test", "embedding loader test", "do these tokens work"]
    tokenizer = get_tokenizer("basic_english")
    tokenized = [tokenizer(doc) for doc in corpus]

    vocab = build_vocab_from_iterator(tokenized, specials=["<unk>", "<pad>"])
    vocab.set_default_index(vocab["<unk>"])

    emb_random = EmbeddingLoader(vocab, config_random).load()
    emb_glove = EmbeddingLoader(vocab, config_glove).load()

    assert emb_random.weight.shape == (len(vocab), 100)
    assert emb_glove.weight.shape == (len(vocab), 100)
    assert isinstance(emb_glove.weight, torch.Tensor)
    assert emb_glove.padding_idx == vocab["<pad>"]
