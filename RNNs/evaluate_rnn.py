import argparse
import os
import torch
import yaml
import json
import pickle
import numpy as np
import pandas as pd
import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, top_k_accuracy_score, confusion_matrix, classification_report

from utils.rnn_preprocessing import RNN_Preprocessor
from utils.dataloader_utils import get_individual_dataloader
from utils.embedding_loader import EmbeddingLoader
from utils.directory_utils import prepare_unique_output_path
from models.rnn_model import RNNClassifier

def load_config(path):
    with open(path, "r") as f:
        return json.load(f) if path.endswith(".json") else yaml.safe_load(f)
    
def evaluate(model, dataloader, criterion, device, label_encoder, test_type, vocab, return_analysis=True):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels, all_outputs = [], [], []
    # final_hidden_states, final_hidden_labels = [], []
    hidden_states, prediction_details, misclassified, attention_examples = [], [], [], []



    with torch.no_grad():
        attn_entropies = []
        attn_max_weights = []
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch)

            if isinstance(output, tuple):
                output, attn_weights = output
                # Mean entropy across all examples
                attn_entropies.append(-(attn_weights * attn_weights.log()).sum(dim=1).mean().item())
                attn_max_weights.append(attn_weights.max().item())

                if return_analysis:
                    # Convert token IDs back to tokens (need vocab reverse map)
                    tokens = [[vocab.vocab.get_itos()[token_id.item()] for token_id in seq] for seq in x_batch]
                    for i in range(len(tokens)):
                        attention_examples.append({
                            "tokens": tokens[i],
                            "attention_weights": attn_weights[i].cpu().tolist()
                        })
            
            preds = torch.argmax(output, dim=1)

            if return_analysis:
                probs = torch.softmax(output, dim=1)
                for i in range(len(y_batch)):
                    prediction_details.append({
                        "true_label": label_encoder.classes_[y_batch[i].item()],
                        "predicted_label": label_encoder.classes_[preds[i].item()],
                        "probabilities": probs[i].cpu().tolist()
                    })

                    if preds[i] != y_batch[i]:
                        misclassified.append({
                            "true_label": label_encoder.classes_[y_batch[i].item()],
                            "predicted_label": label_encoder.classes_[preds[i].item()],
                            "confidence": probs[i][preds[i]].item()
                        })

                # Extract hidden state from final layer
                if hasattr(model, 'final_hidden_state'):  # you may expose this manually
                    h_n = model.final_hidden_state  # shape: [batch, hidden_dim]
                    hidden_states.extend([{
                        "hidden_state": h.cpu().tolist(),
                        "label": label_encoder.classes_[label.item()]
                    } for h, label in zip(h_n, y_batch)])

            
            loss = criterion(output, y_batch)
            #preds = torch.argmax(output, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(y_batch.cpu())
            all_outputs.append(output.cpu())

            total_loss += loss.item() * y_batch.size(0)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
        
    # if final_hidden_states:
    #     hidden_matrix = torch.cat(final_hidden_states).numpy()
    #     hidden_labels = torch.cat(final_hidden_labels).numpy()
    #     np.save(os.path.join(config["output_dir"], "hidden_states.npy"), hidden_matrix)
    #     np.save(os.path.join(config["output_dir"], "hidden_labels.npy"), hidden_labels)


    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_scores = torch.cat(all_outputs).numpy()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "top3_accuracy": top_k_accuracy_score(y_true, y_scores, k=3),
        "test_type": test_type,
        "run_type": "testing"
    }

    if attn_entropies and attn_max_weights:
        metrics["avg_attn_entropy"] = float(np.mean(attn_entropies))
        metrics["avg_attn_max_weight"] = float(np.mean(attn_max_weights))

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

    report = classification_report(
        y_true, y_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )
    metrics["per_class"] = {k: v for k, v in report.items() if k in label_encoder.classes_}

    if return_analysis:
        return total_loss / total, metrics, {
            "misclassified": misclassified,
            "predictions": prediction_details,
            "hidden_states": hidden_states,
            "attention_examples": attention_examples
        }
    return total_loss / total, metrics, None

def main(config_path):
    start_time = time.time()
    config = load_config(config_path)
    setVariables = load_config(os.path.join(config["model_dir"], "constants_for_eval.yaml"))

    for k, v in setVariables.items():
        print(k, " ", v)
        config[k] = v

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[Eval] Using device: {device}")

    with open(os.path.join(config["model_dir"], "vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)

    with open(os.path.join(config["model_dir"], "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)

    test_df = pd.read_csv("../Data/arxiv_test.csv") if not config["test_flight"] else pd.read_csv("../Data/arxiv_test.csv").sample(n=100, random_state=42)
    test_preprocessor = RNN_Preprocessor(test_df, config["preprocess"], vocab=vocab, label_encoder=label_encoder)
    test_preprocessor.preprocess()

    test_df = test_preprocessor.df
    test_df["vocab"] = vocab
    test_df["label_encoder"] = label_encoder

    test_loader = get_individual_dataloader(test_df, batch_size=config["batch_size"], shuffle=False)
    embedding = EmbeddingLoader(vocab, config["embedding"]).load()

    model_cfg = config["model"]
    model = RNNClassifier(
        hidden_dim=model_cfg["hidden_dim"],
        output_dim=len(label_encoder.classes_),
        embedding=embedding,
        rnn_type=model_cfg["rnn_type"],
        bidirectional=model_cfg["bidirectional"],
        attention_layer=model_cfg.get("attention_layer", "None"),
        rtn_attn_weight=model_cfg.get("return_attn_weights", False),
        num_layers=model_cfg.get("num_layers", 1),
        dropout=model_cfg.get("dropout", 0.3),
        batch_first=True
    ).to(device)

    model.load_state_dict(torch.load(f"{config['model_dir']}/best_model.pt"))
    criterion = torch.nn.CrossEntropyLoss()

    test_loss, test_metrics, extras = evaluate(model, test_loader, criterion, device, label_encoder, config["test_type"], vocab)
    print(f"\n[Test Evaluation]")
    print(f"Loss: {test_loss:.4f}")
    for k, v in test_metrics.items():
        if isinstance(v, (int, float)):
            print(f"{k.title()}: {v:.4f}")
        else:
            print(f"{k.title()}: {v}")


    out_dir = config['log_dir']
    with open(f"{out_dir}/test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    flat_row = {
        "accuracy": test_metrics["accuracy"],
        "f1": test_metrics["f1"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "top3_accuracy": test_metrics["top3_accuracy"],
        "avg_attn_entropy": test_metrics.get("avg_attn_entropy", None),
        "avg_attn_max_weight": test_metrics.get("avg_attn_max_weight", None),
        "test_type": test_metrics["test_type"],
        "run_type": "testing"
    }    
    pd.DataFrame([flat_row]).to_csv(f"{out_dir}/test_metrics.csv", index=False)

    # Save all extra logs
    with open(f"{out_dir}/misclassified.json", "w") as f:
        json.dump(extras["misclassified"], f, indent=2)

    with open(f"{out_dir}/prediction_confidences.json", "w") as f:
        json.dump(extras["predictions"], f, indent=2)

    with open(f"{out_dir}/hidden_states.json", "w") as f:
        json.dump(extras["hidden_states"], f, indent=2)

    with open(f"{out_dir}/attention_examples.json", "w") as f:
        json.dump(extras["attention_examples"], f, indent=2)
    
    print(f"\n[Test metrics saved to: {out_dir}]")
    print(f"[Main] Test for {config['test_type']} elapsed for: {time.time() - start_time}s or {(time.time() - start_time) / 60} min")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)