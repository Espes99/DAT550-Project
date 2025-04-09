import os
import json
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

import utils.dataloader_utils as dl
from utils.rnn_preprocessing import RNN_Preprocessor, ArxivDataset
from utils.embedding_loader import EmbeddingLoader
from models.rnn_model import RNNClassifier

class RNNTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"[Device] Using device: {self.device}")

        writer = SummaryWriter(log_dir=config["log_dir"]) if config.get("use_tesorboard", False) else None
        self.metrics = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        self._prepare_data()
        self._init_model()

    def _prepare_data(self):
        raw_train_df = pd.read_csv("../Data/arxiv_train.csv")
        train_preprocessor = RNN_Preprocessor(raw_train_df, self.config["preprocess"])
        train_preprocessor.preprocess()
        full_train_df = train_preprocessor.df
        self.vocab = train_preprocessor.get_vocab()
        self.label_encoder = train_preprocessor.get_label_encoder()

        self.train_df, self.val_df = create_splits(full_train_df)
        self.train_df.attrs["vocab"] = self.vocab
        self.train_df.attrs["label_encoder"] = self.label_encoder
        self.val_df.attrs["vocab"] = self.vocab
        self.val_df.attrs["label_encoder"] = self.label_encoder

        self.df = preprocessor.df
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = pd.read_csv("../Data/arxiv_test.csv")

        self.train_loader, self.val_loader = dl.get_dataloaders_from_splits(self.train_df, self.val_df, batch_size=self.config["batch_size"], shuffle=True)
        self.test_loader = dl.get_individual_dataloader(self.test_df, batch_size=self.config["batch_size"], shuffle=False)
        self.embedding = EmbeddingLoader(preprocessor.get_vocab(), self.config["embedding"]).load()
        self.label_encoder = preprocessor.get_label_encoder()

    def _init_model(self):
        cfg = self.config["model"]
        self.model = RNNClassifier(
            emb_dim=self.config["embedding"]["embedding_dim"],
            hidden_dim=cfg["hidden_dim"],
            output_dim=len(self.label_encoder.classes_),
            embedding=self.embeddinYoug,
            rnn_type=cfg["rnn_type="],
            bidirectional=cfg["bidirectional"],
            attention_layer=cfg.get("attention_layer", "None"),
            rtn_attn_weight=cfg.get("return_attn_weights", False),
            num_layers=cfg.get("num_layers", 1),
            dropout=cfg.get("dropout", 0.3),
            batch_first=True
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["training"]["lr"])


    def train(self):
        best_val_acc = 0

        for epoch in range(self.config["training"]["epochs"]):
            start_time = time.time()
            self.model.train()
            total_loss, correct, total = 0, 0, 0

            for x_batch, y_batch in self.train_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x_batch)
                if isinstance(output, tuple):
                    output = output[0]
                loss = self.criterion(output, y_batch)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * y_batch.size(0)
                preds = torch.argmax(output, dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

            train_acc = correct / total
            avg_train_loss = total_loss / total
            val_acc, val_loss = self.evaluate(self.val_loader)

            self.metrics["train_loss"].append(avg_train_loss)
            self.metrics["val_loss"].append(val_loss)
            self.metrics["train_acc"].append(train_acc)
            self.metrics["val_acc"].append(val_acc)

            if self.writer:
                self.writer.add_scalar("Loss/Train", avg_train_loss, epoch)
                self.writer.add_scalar("Loss/Val", val_loss, epoch)
                self.writer.add_scalar("Accuracy/Train", train_acc, epoch)
                self.writer.add_scalar("Accuracy/Val", val_acc, epoch)

            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1:02d}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f} | Time: {epoch_time:.2f}s")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), os.path.join(self.config["output_dir"], "best_model.pt"))

        if self.writer:
            self.writer.close()
        
        with open(os.path.join(self.config["output_dir"], "metrics.json"), "w") as f:
            json.dump(self.metrics, f, indent=2)
        
        if self.config.get("evaluate_on_test", False):
            self.model.load_state_dict(torch.load(os.path.join(self.config["output_dir"], "best_model.pt")))
            test_acc, test_loss = self.evaluate(self.test_loader)
            print(f"[Test Set] Accuracy: {test_acc:.4f} | Loss: {test_loss:.4f}")
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                output = self.model(x_batch)
                if isinstance(output, tuple):
                    output = output[0]
                loss = self.criterion(output, y_batch)

                total_loss += loss.item() * y_batch.size(0)
                preds = torch.argmax(output, dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
        
        return correct / total, total_loss / total