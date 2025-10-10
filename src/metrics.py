import os
from dataclasses import dataclass
import torch
import math
import json
import numpy as np


@dataclass
class Metrics:
    instances: int = 0
    tokens: int = 0
    correct_tokens: int = 0
    correct_answers: int = 0
    combined_loss: float = 0.0
    token_loss: float = 0.0
    partial_sums_loss: float = 0.0
    correct_ans_tokens: int = 0
    ans_tokens: int = 0

    def to_tensor(self, device, dtype=torch.float32):
        return torch.tensor(
            [
                self.combined_loss,
                self.token_loss,
                self.partial_sums_loss,
                float(self.correct_tokens),
                float(self.tokens),
                float(self.instances),
                float(self.correct_answers),
                float(self.correct_ans_tokens),
                float(self.ans_tokens),
            ],
            dtype=dtype,
            device=device,
        )

    @classmethod
    def from_tensor(cls, t):
        (
            combined_loss,
            token_loss,
            partial_sums_loss,
            correct_tokens,
            tokens,
            instances,
            correct_answers,
            correct_ans_tokens,
            ans_tokens,
        ) = t.tolist()
        return cls(
            instances=int(instances),
            tokens=int(tokens),
            correct_tokens=int(correct_tokens),
            correct_answers=int(correct_answers),
            combined_loss=float(combined_loss),
            token_loss=float(token_loss),
            partial_sums_loss=float(partial_sums_loss),
            correct_ans_tokens=int(correct_ans_tokens),
            ans_tokens=int(ans_tokens),
        )

    def update(self, outputs, batch_size):
        self.combined_loss += outputs.losses.combined_loss.item()
        self.partial_sums_loss += outputs.losses.partial_sums_loss.item()
        self.token_loss += outputs.losses.token_loss.item()
        self.correct_tokens += outputs.acc.total_correct.item()
        self.tokens += outputs.acc.total_tokens
        self.instances += batch_size
        self.correct_answers += outputs.acc.total_correct_answers.item()
        self.correct_ans_tokens += outputs.acc.correct_ans_tokens.item()
        self.ans_tokens += outputs.acc.total_ans_tokens.item()

    def print_metrics_average(self, step, **kwargs):
        try:
            avg_ppl = math.exp(self.token_loss / self.tokens)
        except OverflowError:
            avg_ppl = math.inf
        part_sum_loss = self.partial_sums_loss / self.instances
        extra_metrics = " ".join(f"{k}: {v}" for k, v in kwargs.items())
        print(
            f"Step: {step}. PPL: {avg_ppl}, PartSums: {part_sum_loss:.3f}, {extra_metrics}"
        )


class MetricTracker:
    def __init__(self):
        self.accuracy = {}
        self.token_accuracy = {}
        self.ppl = {}
        self.mse = {}
        self.ans_token_accuracy = {}

    def update(self, metrics, timestep):
        self.accuracy[timestep] = metrics.correct_answers / metrics.instances
        self.token_accuracy[timestep] = metrics.correct_tokens / metrics.tokens
        try:
            ppl = float(math.exp(metrics.token_loss))
        except OverflowError:
            ppl = math.inf
        self.ppl[timestep] = ppl
        self.mse[timestep] = metrics.partial_sums_loss
        self.ans_token_accuracy[timestep] = (
            metrics.correct_ans_tokens / metrics.ans_tokens
        )

    def print_last_metrics(self, name, timestep):
        metrics = [
            f"PPL: {round(self.ppl.get(timestep, 'N/A'), 3)}",
            f"MSE: {round(self.mse.get(timestep, 'N/A'), 3)}",
            f"Accuracy: {round(self.accuracy.get(timestep, 'N/A'), 2)}",
            f"Token Accuracy: {round(self.token_accuracy.get(timestep, 'N/A'), 2)}",
            f"Ans Token Accuracy: {round(self.ans_token_accuracy.get(timestep, 'N/A'), 2)}",
        ]
        print(f"{name} -- " + "; ".join(metrics) + ".")

    def save_as_json(self, path):
        with open(path, "w") as f:
            json.dump(
                {
                    "accuracy": self.accuracy,
                    "token_accuracy": self.token_accuracy,
                    "ppl": self.ppl,
                    "mse": self.mse,
                    "ans_token_accuracy": self.ans_token_accuracy,
                },
                f,
            )


# Convert numpy arrays and tensors to lists for JSON serialization
def convert_for_json(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(item) for item in obj]
    else:
        return obj


def save_metrics(metrics_to_save: dict[str, MetricTracker], save_dir: str):
    """
    Save tracked metrics to disk in both JSON and CSV formats.
    metrics_to_save: mapping from metric name to its MetricTracker
    save_dir: directory path where metric files will be written
    """
    for fname, data in metrics_to_save.items():
        filepath = os.path.join(save_dir, f"{fname}_metric_tracker.json")
        data_to_save = {
            "accuracy": data.accuracy,
            "token_accuracy": data.token_accuracy,
            "ppl": data.ppl,
            "mse": data.mse,
            "ans_token_accuracy": data.ans_token_accuracy,
        }
        with open(filepath, "w") as f:
            data_to_save = convert_for_json(data_to_save)
            json.dump(data_to_save, f)

    print(f"Saved metrics to {save_dir}")


if __name__ == "__main__":
    pass
