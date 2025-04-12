import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation, PillowWriter


def load_attention_trace(directory):
    traces = []
    for fname in sorted(os.listdir(directory), key=lambda x: int(x.split("_")[1].split(".")[0])):
        with open(os.path.join(directory, fname), "r") as f:
            traces.append(json.load(f))
    return traces


def animate_attention(traces, save_path="attention_evolution.gif"):
    tokens = traces[0]["tokens"]
    num_tokens = len(tokens)
    fig, ax = plt.subplots(figsize=(min(12, 0.5 * num_tokens), 2))

    def update(frame):
        ax.clear()
        data = [traces[frame]["attention_weights"]]
        sns.heatmap(
            data,
            cmap="YlGnBu",
            cbar=True,
            xticklabels=tokens,
            yticklabels=[f"Epoch {traces[frame]['epoch'] + 1}"],
            linewidths=0.5,
            linecolor='lightgray',
            ax=ax
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
        ax.set_title("Attention Evolution Across Epochs")

    anim = FuncAnimation(fig, update, frames=len(traces), repeat=True)
    anim.save(save_path, writer=PillowWriter(fps=1))
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_dir", type=str, required=True, help="Directory with attention_trace/*.json")
    parser.add_argument("--out", type=str, default="attention_evolution.gif", help="Output file path")
    args = parser.parse_args()

    traces = load_attention_trace(args.trace_dir)
    animate_attention(traces, args.out)
    print(f"\n[✓] Saved attention animation to: {args.out}")
