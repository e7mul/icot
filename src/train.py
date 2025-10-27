import argparse
import inspect
import json
import os

import torch
import numpy as np

from src.model_utils import create_c_hat_model, save_model_and_optimizer

from src.data_utils import get_loaders
from src.metrics import (
    MetricTracker,
    Metrics,
    save_metrics,
)

from src.transformer import Losses


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
    loss = (
        losses.token_loss
        + p_lambda * losses.partial_sums_loss
        + mse_loss_lambda * losses.mse_output_loss
    )
    return loss


def single_train_loop(
    model, tokenizer, optimizer, train_loader, step, p_lambda, mse_loss_lambda
):
    metrics = Metrics()
    for input_ids, labels, partial_sums in train_loader:
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
            metrics.print_metrics_average(step)
        step += 1

    return metrics, step


def train(args, model, tokenizer, optimizer, datasets):
    step = 0
    best_val_accuracy = float("-inf")
    trackers = {key: MetricTracker() for key in ["train", "val", "test"]}
    for epoch in range(args.epochs):
        model.train()
        train_metrics, step = single_train_loop(
            model,
            tokenizer,
            optimizer,
            datasets.train_loader,
            step,
            args.partial_sums_lambda,
            args.mse_loss_lambda,
        )

        trackers["train"].update(train_metrics, epoch)
        eval_metrics = evaluate(model, tokenizer, datasets.val_loader)
        trackers["val"].update(eval_metrics, epoch)
        trackers["val"].print_last_metrics("Validation", epoch)
        if trackers["val"].accuracy[epoch] > best_val_accuracy:
            best_val_accuracy = trackers["val"].accuracy[epoch]
        if args.test_path:
            test_metrics = evaluate(model, tokenizer, datasets.test_loader)
            trackers["test"].update(test_metrics, epoch)
            trackers["test"].print_last_metrics("Test", epoch)

        save_model_and_optimizer(model, optimizer, args, epoch)
        save_metrics(trackers, args.save_model)


def create_optimizer(args, model):
    # Create Optimizer
    trainable_params = list(model.parameters())
    use_fused = "fused" in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, **extra_args)
    return optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--gpu_ord", type=int, required=True)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_size", type=int, default=-1)
    parser.add_argument("--save_model", type=str, required=True)
    parser.add_argument("--save_config", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--partial_sums_lambda", type=float, default=0.0)
    parser.add_argument("--mse_loss_lambda", type=float, default=0.0)
    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.save_config, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(args.save_config, "args.json"), "w"))

    device = torch.device(f"cuda:{args.gpu_ord}")
    model, tokenizer, config = create_c_hat_model(device)
    paths = {"train": args.train_path, "val": args.val_path, "test": args.test_path}
    datasets = get_loaders(
        paths, max_size=args.max_size, batch_size=args.batch_size, tokenizer=tokenizer
    )
    optimizer = create_optimizer(args, model)
    # save_model_and_optimizer(model, optimizer, args, rank, -1)
    train(args, model, tokenizer, optimizer, datasets)


if __name__ == "__main__":
    main()
