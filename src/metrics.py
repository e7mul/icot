import os
import math
import json

from dataclasses import dataclass, field


@dataclass
class Metrics:
    instances: int = 0
    tokens: int = 0
    correct_tokens: int = 0
    correct_answers: int = 0
    token_loss: float = 0.0
    per_token_ppl: list[float] = field(default_factory=list)
    partial_sums_loss: float = 0.0
    regress_output_loss: float = 0.0
    correct_ans_tokens: int = 0
    ans_tokens: int = 0

    def compute_metrics(self) -> dict[str, float]:
        assert self.instances > 0, "instances must be > 0 to compute averaged metrics"
        assert self.tokens > 0, "tokens must be > 0 to compute token-based metrics"
        assert (
            self.ans_tokens > 0
        ), "ans_tokens must be > 0 to compute ans_token metrics"

        try:
            ppl = float(math.exp(self.token_loss / self.tokens))
        except OverflowError:
            ppl = float("inf")
        return {
            "Acc": self.correct_answers / self.instances,
            "PPL": ppl,
            "PartSums": self.partial_sums_loss / self.instances,
            "RegressOut": self.regress_output_loss / self.instances,
            "AnsTokAcc": self.correct_ans_tokens / self.ans_tokens,
            "PerTokPPL": self.per_token_ppl,
        }

    def format(self, *, step: int | None = None, name: str | None = None) -> str:
        derived_metrics = self.compute_metrics()

        def fmt_val(k, v):
            if isinstance(v, float):
                return f"{k}: {v:.3f}"
            elif isinstance(v, list):
                return f"{k}: {[round(x, 3) for x in v]}"
            else:
                return f"{k}: {v}"

        metrics_str = ", ".join(fmt_val(k, v) for k, v in derived_metrics.items())
        prefix = ""
        if name is not None and step is not None:
            prefix = f"Step: {step}, {name}: "
        elif name is not None:
            prefix = f"{name}: "
        return prefix + metrics_str

    def save_metrics(self, results_dname: str, results_fname: str):
        results_path = os.path.join(
            results_dname, f"{results_fname}_metrics_tracker.json"
        )
        derived_metrics = self.compute_metrics()
        try:
            with open(results_path, "r") as f:
                previous_data = json.load(f)
        except FileNotFoundError:

            previous_data = {}

        for metric, val in derived_metrics.items():
            if metric == "PerTokPPL":
                try:
                    previous_data[metric].append(val)
                except KeyError:
                    previous_data[metric] = [val]
            else:
                previous_data[metric] = previous_data.get(metric, []) + [val]

        os.makedirs(results_dname, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(previous_data, f)

    def parse_to_numbers(self, outputs):
        # Recursively process dataclasses
        def process(obj):
            if hasattr(obj, "__dataclass_fields__"):
                for field in obj.__dataclass_fields__:
                    val = getattr(obj, field)
                    if hasattr(val, "__dataclass_fields__"):
                        process(val)
                    else:
                        try:
                            setattr(obj, field, val.item())
                        except RuntimeError:
                            setattr(obj, field, [i.item() for i in val])
                        except AttributeError:
                            continue

        process(outputs)

    def update(self, outputs, batch_size):
        self.parse_to_numbers(outputs)
        self.partial_sums_loss += outputs.losses.partial_sums_loss
        self.regress_output_loss += outputs.losses.regress_output_loss
        self.token_loss += outputs.losses.token_loss
        self.correct_tokens += outputs.acc.total_correct
        self.tokens += outputs.acc.total_tokens
        self.instances += batch_size
        self.correct_answers += outputs.acc.total_correct_answers
        self.correct_ans_tokens += outputs.acc.correct_ans_tokens
        self.ans_tokens += outputs.acc.total_ans_tokens
        for idx, element in enumerate(outputs.losses.per_token_loss):
            try:
                self.per_token_ppl[idx] += element
            except IndexError:
                self.per_token_ppl.append(element)


if __name__ == "__main__":
    pass
