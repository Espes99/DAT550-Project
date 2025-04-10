import torch.nn as nn
import torch
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchtext")

from torchtext.vocab import GloVe, FastText

class EmbeddingLoader:
    def __init__(self, vocab, config):
        self.vocab = vocab
        self.embedding_dim = config["embedding_dim"]
        self.source = config["embedding_type"]
        self.freeze = config.get("freeze", False)

    def load(self):
        
        if self.source.lower() in ["glove", "fasttext"]:
            vectors = self._load_vectors()
            weights_matrix = self._create_embedding_matrix(vectors)
            return nn.Embedding.from_pretrained(weights_matrix, freeze=self.freeze, padding_idx=self.vocab["<pad>"])
        
        else: # Is intended to catch "random" but setup to also catch strays
            embedding = nn.Embedding(len(self.vocab), self.embedding_dim, padding_idx=self.vocab["<pad>"])
            return embedding
        
    def _load_vectors(self):
        print(f"[EmbeddingLoader] Loading {self.source.upper()} vectors...")
        if self.source == "glove":
            return GloVe(name="6B", dim=self.embedding_dim)
        elif self.source == "fasttext":
            self.embedding_dim = 300
            return FastText(language="en")
        
    def _create_embedding_matrix(self, vectors):
        print("[EmbeddingLoader] Building embedding matrix...")
        matrix = np.zeros((len(self.vocab), self.embedding_dim))
        found = 0

        for idx, token in enumerate(self.vocab.get_itos()):
            if token in vectors.stoi:
                matrix[idx] = vectors[token].numpy()
                found += 1
            else:
                matrix[idx] = np.random.normal(scale=0.6, size=(self.embedding_dim,))
        
        print(f"[EmbeddingLoader] Found {found}/{len(self.vocab)} tokens in {self.source.upper()}.")
        return torch.tensor(matrix, dtype=torch.float32)