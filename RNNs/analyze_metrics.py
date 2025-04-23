import argparse
import os
import json
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.calibration import calibration_curve

from utils.directory_utils import prepare_unique_output_path

def load_csv_or_json(path):
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file type: {path}")

def load_hidden_states(path):
    with open(path, "r") as f:
        data = json.load(f)
    states = np.array([d["hidden_state"] for d in data])
    labels = np.array([d["label"] for d in data])
    return states, labels

def load_predictions(path):
    with open(path, "r") as f:
        return json.load(f)

def load_misclassified(path):
    with open(path, "r") as f:
        return json.load(f)

def load_test_metrics(path):
    with open(path, "r") as f:
        return json.load(f)

def plot_comparative_training_curves(metrics_dfs, labels, save_dir):
    plt.figure()
    for df, label in zip(metrics_dfs, labels):
        plt.plot(df["epoch"], df["val_loss"], label=f"{label} (val)")
        plt.plot(df["epoch"], df["train_loss"], linestyle="--", alpha=0.6, label=f"{label} (train)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_comparison.png"))
    plt.close()

    plt.figure()
    for df, label in zip(metrics_dfs, labels):
        plt.plot(df["epoch"], df["val_acc"], label=f"{label} (val)")
        plt.plot(df["epoch"], df["train_acc"], linestyle="--", alpha=0.6, label=f"{label} (train)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_comparison.png"))
    plt.close()

    if "val_f1" in metrics_dfs[0].columns:
        plt.figure()
        for df, label in zip(metrics_dfs, labels):
            plt.plot(df["epoch"], df["val_f1"], label=f"{label}")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.title("F1 Score Comparison")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "f1_comparison.png"))
        plt.close()

def plot_gradient_norm(metrics_dfs, labels, save_dir):
    plt.figure()
    for df, label in zip(metrics_dfs, labels):
        if "grad_norm" in df.columns:
            plt.plot(df["epoch"], df["grad_norm"], label=f"{label}")
    plt.xlabel("Epoch")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norm Over Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "gradient_norm_comparison.png"))
    plt.close()

def plot_attention_entropy(metrics_dfs, labels, save_dir):
    plt.figure()
    for df, label in zip(metrics_dfs, labels):
        if "val_attn_entropy" in df.columns:
            plt.plot(df["epoch"], df["val_attn_entropy"], label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Entropy")
    plt.title("Validation Attention Entropy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "attention_entropy_comparison.png"))
    plt.close()

def plot_confusion_matrix(conf_matrix, class_labels, label, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix: {label}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix_grid(conf_matrices, class_labels_list, labels, save_path, cols=2):
    n = len(conf_matrices)
    rows = math.ceil(n / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))

    axs = axs.flatten()
    for i, (cm, class_labels, label) in enumerate(zip(conf_matrices, class_labels_list, labels)):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels,
                    yticklabels=class_labels, ax=axs[i])
        axs[i].set_title(f"Confusion Matrix: {label}")
        axs[i].set_xlabel("Predicted")
        axs[i].set_ylabel("True")

    # Hide unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_per_class_metrics(per_class, label, save_path):
    df = pd.DataFrame(per_class).T  # class label as index
    df = df[~df.index.str.contains("avg|accuracy")]
    df[["precision", "recall", "f1-score"]].plot(kind="bar", figsize=(12, 6))
    plt.title(f"Per-Class Metrics: {label}")
    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_per_class_metrics_grid(per_class_metrics_list, labels, save_path, cols=2):
    n = len(per_class_metrics_list)
    rows = math.ceil(n / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 8, rows * 5))
    axs = axs.flatten()

    for i, (per_class, label) in enumerate(zip(per_class_metrics_list, labels)):
        df = pd.DataFrame(per_class).T
        df = df[~df.index.str.contains("avg|accuracy")]
        df[["precision", "recall", "f1-score"]].plot(kind="bar", ax=axs[i])
        axs[i].set_title(f"Per-Class Metrics: {label}")
        axs[i].set_xlabel("Class")
        axs[i].set_ylabel("Score")
        axs[i].tick_params(axis='x', rotation=45)

    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confidence_histogram(predictions, label, save_path):
    correct_conf = []
    wrong_conf = []
    for p in predictions:
        conf = max(p["probabilities"])
        if p["true_label"] == p["predicted_label"]:
            correct_conf.append(conf)
        else:
            wrong_conf.append(conf)
    plt.figure()
    plt.hist([correct_conf, wrong_conf], bins=25, label=["Correct", "Incorrect"], alpha=0.7)
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.title(f"Confidence Histogram: {label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confidence_histogram_grid(predictions_list, labels, save_path, cols=2):
    n = len(predictions_list)
    rows = math.ceil(n / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
    axs = axs.flatten()

    for i, predictions in enumerate(predictions_list):
        label = labels[i]
        correct_conf = []
        wrong_conf = []
        for p in predictions:
            conf = max(p["probabilities"])
            if p["true_label"] == p["predicted_label"]:
                correct_conf.append(conf)
            else:
                wrong_conf.append(conf)
        axs[i].hist([correct_conf, wrong_conf], bins=25, label=["Correct", "Incorrect"], alpha=0.7)
        axs[i].set_title(f"Confidence Histogram: {label}")
        axs[i].set_xlabel("Confidence")
        axs[i].set_ylabel("Count")
        axs[i].legend()

    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_calibration_curve(predictions, label, save_path):
    y_true = [1 if p["true_label"] == p["predicted_label"] else 0 for p in predictions]
    y_probs = [max(p["probabilities"]) for p in predictions]
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"Calibration Curve: {label}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_calibration_curve_grid(predictions_list, labels, save_path, cols=2):
    n = len(predictions_list)
    rows = math.ceil(n / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
    axs = axs.flatten()

    for i, (predictions, label) in enumerate(zip(predictions_list, labels)):
        y_true = [1 if p["true_label"] == p["predicted_label"] else 0 for p in predictions]
        y_probs = [max(p["probabilities"]) for p in predictions]
        prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)
        axs[i].plot(prob_pred, prob_true, marker='o')
        axs[i].plot([0, 1], [0, 1], linestyle='--', color='gray')
        axs[i].set_title(f"Calibration Curve: {label}")
        axs[i].set_xlabel("Confidence")
        axs[i].set_ylabel("Accuracy")

    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def summarize_misclassified(misclassified, label, save_path):
    sorted_samples = sorted(misclassified, key=lambda x: -x["confidence"])[:5]
    with open(save_path, "w") as f:
        f.write(f"Top 5 Most Confident Misclassifications for {label}\n\n")
        for sample in sorted_samples:
            f.write(json.dumps(sample, indent=2) + "\n\n")

def visualize_hidden_states_per_model(hidden_data, method="tsne", save_dir=".", tag=""):
    reducer = TSNE(n_components=2, perplexity=30, n_iter=500, learning_rate="auto", init="pca") if method == "tsne" else PCA(n_components=2)
    for label, states, class_labels in hidden_data:
        reduced = reducer.fit_transform(states)
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=pd.factorize(class_labels)[0], cmap='tab10', alpha=0.7)
        plt.colorbar(scatter, label="Class")
        plt.title(f"{method.upper()} of Hidden States ({label})")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{tag}_{label}_{method}.png"))
        plt.close()

def plot_hidden_states_grid(hidden_data, method="pca", save_path="hidden_states_grid.png"):
    assert method in ["pca", "tsne"], "method must be 'pca' or 'tsne'"

    reducer = PCA(n_components=2) if method == "pca" else TSNE(
        n_components=2, perplexity=30, n_iter=500, learning_rate="auto", init="pca"
    )

    num_models = len(hidden_data)
    cols = 2
    rows = math.ceil(num_models / cols)

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
    axs = axs.flatten()

    for i, (label, states, class_labels) in enumerate(hidden_data):
        try:
            reduced = reducer.fit_transform(states)
            axs[i].scatter(
                reduced[:, 0],
                reduced[:, 1],
                c=pd.factorize(class_labels)[0],
                cmap="tab10",
                alpha=0.7,
                s=15
            )
            axs[i].set_title(label)
            axs[i].set_xlabel("Component 1")
            axs[i].set_ylabel("Component 2")
        except Exception as e:
            axs[i].text(0.5, 0.5, f"Error in {label}\n{str(e)}", ha='center', va='center')
            axs[i].set_axis_off()

    # Hide any unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].set_axis_off()

    fig.suptitle(f"{method.upper()} Projection of Hidden States", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()

def plot_training_metrics_grid(metrics_dfs, labels, save_path):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()

    # Loss
    for df, label in zip(metrics_dfs, labels):
        axs[0].plot(df["epoch"], df["val_loss"], label=f"{label} (val)")
        axs[0].plot(df["epoch"], df["train_loss"], linestyle="--", alpha=0.6, label=f"{label} (train)")
    axs[0].set_title("Loss Curve")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    # Accuracy
    for df, label in zip(metrics_dfs, labels):
        axs[1].plot(df["epoch"], df["val_acc"], label=f"{label} (val)")
        axs[1].plot(df["epoch"], df["train_acc"], linestyle="--", alpha=0.6, label=f"{label} (train)")
    axs[1].set_title("Accuracy Curve")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()

    # F1 Score
    for df, label in zip(metrics_dfs, labels):
        if "val_f1" in df.columns:
            axs[2].plot(df["epoch"], df["val_f1"], label=label)
    axs[2].set_title("F1 Score (Validation)")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("F1 Score")
    axs[2].legend()

    # Gradient Norm
    for df, label in zip(metrics_dfs, labels):
        if "grad_norm" in df.columns:
            axs[3].plot(df["epoch"], df["grad_norm"], label=label)
    axs[3].set_title("Gradient Norm")
    axs[3].set_xlabel("Epoch")
    axs[3].set_ylabel("Norm")
    axs[3].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def generate_summary_csv(test_metrics_list, labels, save_path):
    rows = []
    for metrics, label in zip(test_metrics_list, labels):
        row = {
            "Model": label,
            "Accuracy": round(metrics.get("accuracy", 0), 4),
            "Macro-F1": round(metrics.get("f1", 0), 4),
            "Precision": round(metrics.get("precision", 0), 4),
            "Recall": round(metrics.get("recall", 0), 4),
            "Top-3 Accuracy": round(metrics.get("top3_accuracy", 0), 4),
            "Avg Attention Entropy": round(metrics.get("avg_attn_entropy", 0), 4) if "avg_attn_entropy" in metrics else "N/A",
            "Loss": round(metrics.get("loss", 0), 4) if "loss" in metrics else "N/A",
            "Test Type": metrics.get("test_type", "Unknown"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"[✓] Exported summary CSV to: {save_path}")

def plot_test_metric_summary(test_metrics_list, labels, save_path):
    metrics = ["accuracy", "precision", "recall", "f1", "top3_accuracy"]
    df = pd.DataFrame([
        {"Model": label, "Metric": metric.title(), "Value": m.get(metric, 0)}
        for label, m in zip(labels, test_metrics_list)
        for metric in metrics
    ])

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Model", y="Value", hue="Metric", palette="pastel")
    plt.title("Test Metric Comparison Across Models")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.xticks(rotation=45, ha="right")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_test_metric_summary_heatmap(test_metrics_list, labels, save_path):
    metrics = ["accuracy", "precision", "recall", "f1", "top3_accuracy"]
    data = [[m.get(metric, 0) for metric in metrics] for m in test_metrics_list]
    df = pd.DataFrame(data, index=labels, columns=[m.title() for m in metrics])

    plt.figure(figsize=(10, len(labels) * 0.6 + 2))
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".2f", vmin=0, vmax=1)
    plt.title("Test Metrics Heatmap")
    plt.ylabel("Model")
    plt.xlabel("Metric")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main(run_dirs, mode="both"):
    save_dir = prepare_unique_output_path(os.path.join("analysis", "comparison", "testing"))
    metrics_dfs = []
    hidden_data = []
    labels = []
    if len(run_dirs) > 4:
        print(f"[Error] You selected more than 4 models")
        return
    
    conf_matrix_list, calibration_curve_list, per_class_metrics_list, conf_hist = (
        {"data": [], "class_label": [], "label": []}, {"data": [], "class_label": [], "label": []}, {"data": [], "class_label": [], "label": []}, {"data": [], "class_label": [], "label": []}
    )

    test_metrics_list = []

    for run_dir in run_dirs:
        label = os.path.basename(run_dir)
        metrics_path = os.path.join(run_dir, "metrics.csv")
        hidden_path = os.path.join(run_dir, "hidden_states.json")
        predictions_path = os.path.join(run_dir, "prediction_confidences.json")
        misclassified_path = os.path.join(run_dir, "misclassified.json")
        test_metrics_path = os.path.join(run_dir, "test_metrics.json")

        test_metrics_path = os.path.join(run_dir, "test_metrics.json")
        if os.path.exists(test_metrics_path):
            test_metrics_list.append(load_test_metrics(test_metrics_path))
        else:
            print(f"[Warning] No test_metrics.json found in {run_dir}")

        metrics_df = load_csv_or_json(metrics_path)
        metrics_dfs.append(metrics_df)
        labels.append(label)

        if mode in ["train", "both"]:
            if os.path.exists(hidden_path):
                states, lbls = load_hidden_states(hidden_path)
                hidden_data.append((label, states, lbls))

        if mode in ["test", "both"]:
            if os.path.exists(predictions_path):
                predictions = load_predictions(predictions_path)
                conf_hist["data"].append(predictions)
                conf_hist["label"].append(label)
                calibration_curve_list["data"].append(predictions)
                calibration_curve_list["label"].append(label)
                #plot_confidence_histogram(predictions, label, os.path.join(save_dir, f"confidence_hist_{label}.png"))
                #plot_calibration_curve(predictions, label, os.path.join(save_dir, f"calibration_curve_{label}.png"))

            if os.path.exists(misclassified_path):
                misclassified = load_misclassified(misclassified_path)
                summarize_misclassified(misclassified, label, os.path.join(save_dir, f"misclassified_{label}.txt"))

            if os.path.exists(test_metrics_path):
                test_metrics = load_test_metrics(test_metrics_path)
                conf_matrix_list["data"].append(test_metrics["confusion_matrix"])
                if "confusion_matrix" in test_metrics:
                    class_labels = list(test_metrics.get("per_class", {}).keys())
                    conf_matrix_list["class_label"].append(class_labels)
                    conf_matrix_list["label"].append(label)
                    #plot_confusion_matrix(test_metrics["confusion_matrix"], class_labels, label, os.path.join(save_dir, f"confusion_matrix_{label}.png"))
                if "per_class" in test_metrics:
                    per_class_metrics_list["data"].append(test_metrics["per_class"])
                    per_class_metrics_list["label"].append(label)
                    #plot_per_class_metrics(test_metrics["per_class"], label, os.path.join(save_dir, f"per_class_metrics_{label}.png"))

    if mode in ["train", "both"]:
        #plot_comparative_training_curves(metrics_dfs, labels, save_dir)
        plot_attention_entropy(metrics_dfs, labels, save_dir)
        #plot_gradient_norm(metrics_dfs, labels, save_dir)
        #visualize_hidden_states_per_model(hidden_data, method="tsne", save_dir=save_dir, tag="hidden_states")
        #visualize_hidden_states_per_model(hidden_data, method="pca", save_dir=save_dir, tag="hidden_states")
        plot_hidden_states_grid(hidden_data, method="tsne", save_path=os.path.join(save_dir, "hidden_states_tsne.png"))
        plot_hidden_states_grid(hidden_data, method="pca", save_path=os.path.join(save_dir, "hidden_states_pca.png"))
        plot_training_metrics_grid(
            metrics_dfs, 
            labels, 
            os.path.join(save_dir, "training_performance_overview.png")
        )

    if mode in ["test", "both"]:
        plot_test_metric_summary_heatmap(test_metrics_list, labels, os.path.join(save_dir, "test_metric_comparison_heatmap.png"))
        plot_test_metric_summary(test_metrics_list, labels, os.path.join(save_dir, "test_metric_comparison.png"))
        plot_confidence_histogram_grid(conf_hist["data"], conf_hist["label"], os.path.join(save_dir, "confidence_comparative_histogram.png"))
        plot_calibration_curve_grid(calibration_curve_list["data"], calibration_curve_list["label"], os.path.join(save_dir, "calibration_comparative_curve.png"))
        plot_confusion_matrix_grid(conf_matrix_list["data"], conf_matrix_list["class_label"], conf_matrix_list["label"], os.path.join(save_dir, "confusion_matrix_comparative.png"))
        plot_per_class_metrics_grid(per_class_metrics_list["data"], per_class_metrics_list["label"], os.path.join(save_dir, "per_class_metrics_comparative.png"))

    summary_csv_path = os.path.join(save_dir, "model_comparison_summary.csv")
    generate_summary_csv(test_metrics_list, labels, summary_csv_path)
    print(f"\n[✓] Saved all plots and summaries to: {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="List of run directories to analyze")
    parser.add_argument("--mode", type=str, default="both", choices=["train", "test", "both"], help="Which type of analysis to run")
    args = parser.parse_args()
    main(args.runs, mode=args.mode)
