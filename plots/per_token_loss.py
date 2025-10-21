import os
import json
import matplotlib.pyplot as plt
import numpy as np

rpath = "results/4_by_4_mult"

D = 2 * int(rpath.split("/")[-1].split("_")[0])

for dirname in os.listdir(rpath):

    with open(
        os.path.join(rpath, dirname, "checkpoints/val_metric_tracker.json"), "r"
    ) as f:
        data = json.load(f)

    per_token_loss = data.get("per_token_loss", {})

    # Convert string keys to ints and sort
    steps = sorted([int(k) for k in per_token_loss.keys()])
    # Get max token length from any entry
    max_token_len = max(len(v) for v in per_token_loss.values())

    # Build an array of shape [num_tokens, num_steps]
    # Each row: step; each col: token position
    loss_matrix = np.zeros((len(steps), max_token_len))
    for i, step in enumerate(steps):
        losses = per_token_loss[str(step)]
        loss_matrix[i, : len(losses)] = losses

    # Only plot token positions for which the initial value (step index 0) is above a small threshold
    tolerance = 1e-8
    token_positions_to_plot = np.arange(max_token_len)[-D:]
    plt.figure(figsize=(10, 6))
    for token_pos in token_positions_to_plot:
        plt.plot(
            steps,
            loss_matrix[:, token_pos],
            label=f"Token {token_pos}",
            color=plt.cm.viridis(token_pos / max_token_len),
        )

    plt.xlabel("Step")
    plt.ylabel("Per Token Loss")
    plt.title("Per Token Loss vs Step (Colored by Token Position)")
    plt.legend(title="Token Position", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(rpath, dirname, "per_token_loss_across_steps.png"))
