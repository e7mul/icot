import argparse
import inspect
import json
import os
from dataclasses import dataclass, field

import torch
import numpy as np

from src.model_utils import (
    create_c_hat_model,
    load_c_hat_model,
    save_checkpoint,
)


from src.data_utils import get_loaders
from src.metrics import Metrics
from src.transformer import Losses


@dataclass
class Paths:
    train_fname: str
    results_dname: str

    val_fname: str | None = None
    test_fname: str | None = None
    ckpt_fname: str | None = None
    optim_fname: str | None = None

    ckpt_dname: str = field(init=False)
    optim_dname: str = field(init=False)
    trackers_dname: str = field(init=False)

    def __post_init__(self) -> None:
        self.ckpt_dname = os.path.join(self.results_dname, "checkpoints")
        self.optim_dname = os.path.join(self.results_dname, "optimizers")
        self.trackers_dname = os.path.join(self.results_dname, "trackers")

    @classmethod
    def from_args(cls, args):
        # Only keep arguments relevant to Paths constructor
        valid_keys = inspect.signature(cls).parameters.keys()
        args_dict = {k: v for k, v in vars(args).items() if k in valid_keys}
        ckpt_fname = args_dict.get("ckpt_fname", None)
        if ckpt_fname is not None:
            args_dict["optim_fname"] = ckpt_fname.replace("checkpoint", "optimizer")
        return cls(**args_dict)

    def __repr__(self) -> str:
        attrs = [
            f"train_fname={repr(self.train_fname)}",
            f"results_dname={repr(self.results_dname)}",
            f"val_fname={repr(self.val_fname)}",
            f"test_fname={repr(self.test_fname)}",
            f"ckpt_fname={repr(self.ckpt_fname)}",
            f"optim_fname={repr(self.optim_fname)}",
            f"ckpt_dname={repr(self.ckpt_dname)}",
            f"optim_dname={repr(self.optim_dname)}",
            f"trackers_dname={repr(self.trackers_dname)}",
        ]
        return f"Paths(\n  " + ",\n  ".join(attrs) + "\n)"


def get_starting_epoch(ckpt_fname: str) -> int:
    """
    The function assumes that ckpt_fname is of the form dir1/dir2/.../checkpoints/checkpoint_X.pt and returns int(X)
    """
    base = os.path.basename(ckpt_fname)
    # Expect base to be 'checkpoint_X.pt'
    name, ext = os.path.splitext(base)
    if not name.startswith("checkpoint_"):
        raise ValueError(f"Unexpected checkpoint filename: {ckpt_fname}")
    num_str = name.replace("checkpoint_", "")
    try:
        return int(num_str) + 1
    except ValueError:
        raise ValueError(
            f"Could not extract epoch number from checkpoint filename: {ckpt_fname}"
        )


@torch.no_grad()
def evaluate(model, tokenizer, dataloader):
    model.eval()
    metrics = Metrics()
    for input_ids, labels, partial_sums in dataloader:
        sep_pos = get_sep_position(tokenizer, input_ids)
        input_ids = input_ids.to(model.device)
        labels = labels.to(model.device)
        partial_sums = partial_sums.to(model.device)
        batch_size = len(input_ids)
        outputs = model.compute_loss(input_ids, labels, partial_sums, sep_pos)
        metrics.update(outputs, batch_size)
    return metrics


def get_sep_position(tokenizer, input_ids):
    first_sample = input_ids[0].to("cpu")
    sep_id = tokenizer("###")[f"input_ids"][0]
    sep_pos = np.where(first_sample == sep_id)[0][0]
    assert (input_ids[:, sep_pos] == sep_id).all(), "Inconsistent separator positions"
    return sep_pos


def combine_losses(
    losses: Losses, p_lambda: float, mse_loss_lambda: float
) -> torch.Tensor:
    """Compute the weighted combination of loss components.

    Args:
        losses: Object containing token_loss, partial_sums_loss, and mse_output_loss
        p_lambda: Weight for the partial sums loss component
        mse_loss_lambda: Weight for the MSE output loss component

    Returns:
        Combined loss as a scalar tensor
    """
    loss = (
        losses.token_loss
        + p_lambda * losses.partial_sums_loss
        + mse_loss_lambda * losses.regress_output_loss
    )
    return loss


def single_train_loop(model, tokenizer, optimizer, loader, p_lambda, mse_loss_lambda):
    metrics = Metrics()
    for step, (input_ids, labels, partial_sums) in enumerate(loader):
        sep_pos = get_sep_position(tokenizer, input_ids)
        input_ids = input_ids.to(model.device)
        labels = labels.to(model.device)
        partial_sums = partial_sums.to(model.device)
        batch_size = len(input_ids)
        outputs = model.compute_loss(
            input_ids, labels, partial_sums, separator_position=sep_pos
        )
        loss = combine_losses(outputs.losses, p_lambda, mse_loss_lambda)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        metrics.update(outputs, batch_size)

        if step % 100 == 0:
            print(metrics.format(step=step, name="Train"))
    return metrics


def train(args, model, tokenizer, optimizer, datasets, paths):
    if paths.ckpt_fname is not None:
        start_epoch = get_starting_epoch(paths.ckpt_fname)
    else:
        start_epoch = 0
    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        train_metrics = single_train_loop(
            model,
            tokenizer,
            optimizer,
            datasets.train_loader,
            args.partial_sums_lambda,
            args.mse_loss_lambda,
        )
        train_metrics.save_metrics(paths.trackers_dname, "train")

        eval_metrics = evaluate(model, tokenizer, datasets.val_loader)
        print(eval_metrics.format(step=epoch, name="Val"))
        eval_metrics.save_metrics(paths.trackers_dname, "val")
        if paths.test_fname:
            test_metrics = evaluate(model, tokenizer, datasets.test_loader)
            print(test_metrics.format(step=epoch, name="Test"))
            test_metrics.save_metrics(paths.trackers_dname, "test")
        save_checkpoint(model, paths.ckpt_dname, epoch)
        save_checkpoint(optimizer, paths.optim_dname, epoch)


def create_optimizer(args, model):
    # Create Optimizer
    trainable_params = list(model.parameters())
    use_fused = "fused" in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, **extra_args)
    return optimizer


def load_optimizer(
    optimizer: torch.optim.Optimizer, optim_path: str
) -> torch.optim.Optimizer:
    """
    Loads optimizer state from the specified path into the provided optimizer.
    """
    state_dict = torch.load(optim_path, map_location="cpu")
    optimizer.load_state_dict(state_dict)
    return optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_fname", type=str, required=True)
    parser.add_argument("--val_fname", type=str, required=True)
    parser.add_argument("--test_fname", type=str, default=None)
    parser.add_argument(
        "--results_dname", type=str, required=True, help="Directory for results"
    )
    parser.add_argument(
        "--ckpt_fname",
        type=str,
        default=None,
        help="Path to checkpoint for loading state dict",
    )
    parser.add_argument("--gpu_ord", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_size", type=int, default=-1)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--partial_sums_lambda", type=float, default=0.0)
    parser.add_argument("--mse_loss_lambda", type=float, default=0.0)
    args = parser.parse_args()

    paths = Paths.from_args(args)

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(paths.results_dname, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(paths.results_dname, "args.json"), "w"))

    device = torch.device(f"cuda:{args.gpu_ord}")
    model, tokenizer, config = create_c_hat_model(device)
    optimizer = create_optimizer(args, model)

    if paths.ckpt_fname:
        model, tokenizer = load_c_hat_model(paths.ckpt_fname, device)
        optimizer = load_optimizer(optimizer, paths.optim_fname)

    datasets = get_loaders(
        paths, max_size=args.max_size, batch_size=args.batch_size, tokenizer=tokenizer
    )
    train(args, model, tokenizer, optimizer, datasets, paths)


if __name__ == "__main__":
    main()
