import os
import json
import time
import pickle
import yaml

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, precision_score, recall_score, top_k_accuracy_score

import utils.dataloader_utils as dl
from utils.rnn_preprocessing import RNN_Preprocessor
from utils.embedding_loader import EmbeddingLoader
from utils.directory_utils import prepare_unique_output_path
from models.rnn_model import RNNClassifier
torch.backends.cudnn.benchmark = True

class RNNTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"[Device] Using device: {self.device}")

        self.writer = SummaryWriter(os.path.join(config["log_dir"], config["test_type"])) if config.get("use_tensorboard", False) else None
        self.metrics = []

        self.config["output_dir"] = prepare_unique_output_path(os.path.join(config["output_dir"], config["test_type"]))
        self.config["log_dir"] = prepare_unique_output_path(os.path.join(config["log_dir"], config["test_type"]))

        self.save_training_constants(config["output_dir"])
        self.attn_save_dir = os.path.join(self.config["output_dir"], "attention_trace")
        os.makedirs(self.attn_save_dir, exist_ok=True)


        self._prepare_data()
        self._init_model()

    def _prepare_data(self):
        if self.config["test_flight"]:
            raw_train_df = pd.read_csv("../Data/arxiv_train.csv").sample(n=100, random_state=42)
            raw_val_df = pd.read_csv("../Data/arxiv_val.csv").sample(n=100, random_state=42)
        else:
            raw_train_df = pd.read_csv("../Data/arxiv_train.csv")
            raw_val_df = pd.read_csv("../Data/arxiv_val.csv")
        train_preprocessor = RNN_Preprocessor(raw_train_df, self.config["preprocess"])
        train_preprocessor.preprocess()
        val_preprocessor = RNN_Preprocessor(raw_val_df, self.config["preprocess"])
        val_preprocessor.preprocess()
        self.vocab = train_preprocessor.get_vocab()
        self.label_encoder = train_preprocessor.get_label_encoder()

        self.train_df = train_preprocessor.df
        self.train_df.attrs["vocab"] = self.vocab
        self.train_df.attrs["label_encoder"] = self.label_encoder

        self.val_df =  val_preprocessor.df
        self.val_df.attrs["vocab"] = val_preprocessor.vocab
        self.val_df.attrs["label_encoder"] = val_preprocessor.label_encoder

        raw_test_df = pd.read_csv("../Data/arxiv_test.csv") if self.config["test_flight"] else pd.read_csv("../Data/arxiv_test.csv").sample(n=100, random_state=42)
        test_preprocessor = RNN_Preprocessor(raw_test_df, self.config["preprocess"])
        test_preprocessor.preprocess()

        self.test_df = test_preprocessor.df

        self.train_loader, self.val_loader = dl.get_dataloaders_from_splits(self.train_df, self.val_df, batch_size=self.config["batch_size"], shuffle=True)
        self.test_loader = dl.get_individual_dataloader(self.test_df, batch_size=self.config["batch_size"], shuffle=False)
        self.embedding = EmbeddingLoader(train_preprocessor.get_vocab(), self.config["embedding"]).load()
        self.label_encoder = train_preprocessor.get_label_encoder()

        with open(os.path.join(self.config["output_dir"], "vocab.pkl"), "wb") as f:
            pickle.dump(train_preprocessor.get_vocab(), f)

        with open(os.path.join(self.config["output_dir"], "label_encoder.pkl"), "wb") as f:
            pickle.dump(train_preprocessor.get_label_encoder(), f)

    def _init_model(self):
        cfg = self.config["model"]
        self.model = RNNClassifier(
            hidden_dim=cfg["hidden_dim"],
            output_dim=len(self.label_encoder.classes_),
            embedding=self.embedding,
            rnn_type=cfg["rnn_type"],
            bidirectional=cfg["bidirectional"],
            attention_layer=cfg.get("attention_layer", "None"),
            rtn_attn_weight=cfg.get("return_attn_weights", False),
            num_layers=cfg.get("num_layers", 1),
            dropout=cfg.get("dropout", 0.3),
            batch_first=True
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["training"]["lr"])

        sched_cfg = self.config["training"].get("scheduler", None)
        if sched_cfg and sched_cfg["name"].lower() == "ReduceLROnPlateau".lower():
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=sched_cfg.get("mode", "max"),
                factor=sched_cfg.get("factor", 0.5),
                patience=sched_cfg.get("patience", 2),
                verbose=sched_cfg.get("verbose", False)
            )
        else:
            self.scheduler = None



    def train(self):
        start_time = time.time()
        print("[RNNTrainer]: Starting train...")
        best_val_acc = 0
        early_stop_counter = 0
        patience = self.config["training"].get("early_stopping_patience", 3)
        use_early_stop = self.config["training"].get("early_stopping", True)

        for epoch in range(self.config["training"]["epochs"]):
            print(f"[RNNTrainer]: Starting train on epoch {epoch}")
            # Select fixed example only once for tracking
            if epoch == 0:
                x_fixed, y_fixed = next(iter(self.val_loader))
                self.example_tokens = [self.vocab.get_itos()[i.item()] for i in x_fixed[0]]
                self.fixed_input = x_fixed[0].unsqueeze(0).to(self.device)

            epoch_start_time = time.time()

            train_acc, avg_train_loss, grad_norm, train_attn_entropy, train_max_attn = self._train_epoch()

            # Constants to limit token/attention length
            MAX_ATTENTION_TOKENS = 40

            if self.config["model"].get("return_attn_weights", False):
                self.model.eval()
                with torch.no_grad():
                    output = self.model(self.fixed_input)
                    if isinstance(output, tuple):
                        _, attn_weights = output
                        attn_weights = attn_weights[0].cpu().tolist()
                        
                        # Filter to max tokens
                        truncated_tokens = self.example_tokens[:MAX_ATTENTION_TOKENS]
                        truncated_weights = attn_weights[:MAX_ATTENTION_TOKENS]

                        attn_log = {
                            "epoch": epoch,
                            "tokens": truncated_tokens,
                            "attention_weights": truncated_weights
                        }

                        with open(os.path.join(self.attn_save_dir, f"epoch_{epoch}.json"), "w") as f:
                            json.dump(attn_log, f, indent=2)

            val_acc, val_loss, f1, precision, recall, top3, val_attn_entropy, val_max_attn = self._evaluate_epoch(self.val_loader)



            if self.writer:
                self.writer.add_scalar("Loss/Train", avg_train_loss, epoch)
                self.writer.add_scalar("Loss/Val", val_loss, epoch)
                self.writer.add_scalar("Accuracy/Train", train_acc, epoch)
                self.writer.add_scalar("Accuracy/Val", val_acc, epoch)
                self.writer.add_scalar("Grad norm", grad_norm, epoch)
                self.writer.add_scalar("Train attention entropy", train_attn_entropy, epoch)
                self.writer.add_scalar("Train attention max", train_max_attn, epoch)
                self.writer.add_scalar("Validation attention entropy", val_attn_entropy, epoch)
                self.writer.add_scalar("Validation attention max", val_max_attn, epoch)
                self.writer.add_scalar("Validation F1", f1, epoch)
                self.writer.add_scalar("Validation Precision", precision, epoch)
                self.writer.add_scalar("Validation Recall", recall, epoch)
                self.writer.add_scalar("Validation top 3 accuracy", top3, epoch)

            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1:02d}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f} | Time: {epoch_time:.2f}s")

            if self.scheduler:
                self.scheduler.step(val_acc)

                if self.writer:
                    self.writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)
            
                print(f"Current LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            self.metrics.append({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "lr": self.optimizer.param_groups[0]["lr"],
                "grad_norm": grad_norm,
                "train_attn_entropy": train_attn_entropy,
                "train_attn_max": train_max_attn,
                "val_attn_entropy": val_attn_entropy,
                "val_max_attn": val_max_attn,
                "val_f1": f1,
                "val_precision": precision,
                "val_recall": recall,
                "val_top3_acc": top3,
                "test_type": self.config["test_type"],
                "run_type": self.config["run_type"]
            })

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), os.path.join(self.config["output_dir"], "best_model.pt"))
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if use_early_stop and early_stop_counter >= patience:
                    print("Early stopping triggered.")
                    break

        if self.writer:
            self.writer.close()

        with open(os.path.join(self.config["output_dir"], "metrics.json"), "w") as f:
            json.dump(self.metrics, f, indent=2)

        metrics_df = pd.DataFrame(self.metrics)
        metrics_df.to_csv(os.path.join(self.config["output_dir"], "metrics.csv"), index=False)

        with open(os.path.join(self.config["output_dir"], "model_configurations.yaml"), "w") as f:
            yaml.dump(self.config)

        elapsed = time.time() - start_time
        print(f"[Main] Test for {self.config['test_type']} elapsed for: {elapsed:.2f}s or {elapsed / 60:.2f} min")


        if self.config.get("evaluate_on_test", False):
            self.model.load_state_dict(torch.load(os.path.join(self.config["output_dir"], "best_model.pt")))
            test_acc, test_loss = self._evaluate_epoch(self.test_loader)
            print(f"[Test Set] Accuracy: {test_acc:.4f} | Loss: {test_loss:.4f}")


    def _train_epoch(self):
        self.model.train()
        total_loss, correct, total = 0, 0, 0

        for x_batch, y_batch in self.train_loader:
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(x_batch)

            if isinstance(output, tuple):  # e.g., (logits, attention_weights) TODO: Store output[1] for analysis later
                output, attn_weights = output
                # Mean entropy across all examples
                attn_entropy = -(attn_weights * attn_weights.log()).sum(dim=1).mean().item()
                max_attn = attn_weights.max().item()
            else:
                attn_entropy = None
                max_attn = None

            loss = self.criterion(output, y_batch)
            loss.backward()
            grad_norm = sum(p.grad.norm().item() for p in self.model.parameters() if p.grad is not None)
            self.optimizer.step()

            total_loss += loss.item() * y_batch.size(0)
            preds = torch.argmax(output, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        train_acc = correct / total
        avg_train_loss = total_loss / total
        return train_acc, avg_train_loss, grad_norm, attn_entropy, max_attn

    def _evaluate_epoch(self, dataloader):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        all_preds, all_labels, all_outputs = [], [], []


        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                output = self.model(x_batch)

                if isinstance(output, tuple):  # handles models returning attention weights TODO: Store output[1] for analysis later
                    output, attn_weights = output
                    # Mean entropy across all examples
                    attn_entropy = -(attn_weights * attn_weights.log()).sum(dim=1).mean().item()
                    max_attn = attn_weights.max().item()
                else:
                    attn_entropy = None
                    max_attn = None

                loss = self.criterion(output, y_batch)

                total_loss += loss.item() * y_batch.size(0)
                preds = torch.argmax(output, dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(y_batch.cpu())
                all_outputs.append(output.cpu())
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        y_true = torch.cat(all_labels).numpy()
        y_pred = torch.cat(all_preds).numpy()
        y_scores = torch.cat(all_outputs).numpy()

        f1 = f1_score(y_true, y_pred, average="macro")
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        top3 = top_k_accuracy_score(y_true, y_scores, k=3)

        acc = correct / total
        avg_loss = total_loss / total
        return acc, avg_loss, f1, precision, recall, top3, attn_entropy, max_attn
    
    def save_training_constants(self, output_dir, filename="constants_for_eval.yaml"):
        # These are the config sections that must match during evaluation
        keys_to_keep = ["model", "embedding", "preprocess", "test_type", "run_type"]

        # Build dictionary with only relevant keys
        filtered_config = {k: v for k, v in self.config.items() if k in keys_to_keep}

        # Save as YAML next to best_model.pt
        full_path = os.path.join(output_dir, filename)
        with open(full_path, "w") as f:
            yaml.dump(filtered_config, f, default_flow_style=False)
        
        full_path