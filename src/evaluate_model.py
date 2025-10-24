import argparse
import torch

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from jaxtyping import Integer, Float, Array

from src.model_utils import load_c_hat_model
from src.train import get_sep_position
from src.data_utils import PSDataset


def greedy_search(
    model: torch.nn.Module,
    input_ids: Integer[Array, "n_samples seq_len"],
) -> Integer[Array, "n_samples 1"]:
    output = model.forward(input_ids)
    logits = output.logits
    selected_tokens = logits[:, -1, :].argmax(dim=-1).view(-1, 1)
    return selected_tokens


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    input_ids: Integer[Array, "n_samples seq_len"],
    max_new_tokens: int,
) -> Integer[Array, "n_samples seq_len"]:

    model.eval()
    sep_position = get_sep_position(tokenizer, input_ids)
    generated_ids = input_ids[:, : sep_position + 1]
    for i in range(max_new_tokens):
        selected_tokens = greedy_search(model, generated_ids)
        generated_ids = torch.cat((generated_ids, selected_tokens), dim=1)
    return generated_ids


def get_data(
    data_path: str, tokenizer: PreTrainedTokenizerBase
) -> Integer[Array, "n_samples seq_len"]:
    dataset = PSDataset(tokenizer, data_path)
    return torch.tensor(dataset.examples_all, dtype=torch.long)


def compute_per_token_accuracy(
    output: Integer[Array, "n_samples seq_len"],
    target: Integer[Array, "n_samples seq_len"],
    tokenizer: PreTrainedTokenizerBase,
) -> Integer[Array, "output_len"]:
    sep_position = get_sep_position(tokenizer, target)
    output = output[:, sep_position + 1 :]
    target = target[:, sep_position + 1 :]
    return (output == target).float().mean(dim=0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--gpu_ord", type=int, required=False, default=0)
    parser.add_argument("--max_new_tokens", type=int, required=False, default=None)
    args = parser.parse_args()

    if args.max_new_tokens is None:
        # Fallback to parsing from data_path if not provided
        try:
            max_new_tokens = 2 * int(args.data_path.split("/")[1].split("_")[0])
        except (IndexError, ValueError):
            raise ValueError(
                "Could not parse max_new_tokens from data_path. Please provide --max_new_tokens explicitly."
            )
    else:
        max_new_tokens = args.max_new_tokens
    state_dict_path = args.ckpt_path
    max_size = 1000
    batch_size = 32

    device = torch.device(f"cuda:{args.gpu_ord}")
    model, tokenizer = load_c_hat_model(state_dict_path, device)

    input_ids = get_data(args.data_path, tokenizer).to(device)

    model_output = generate(
        model,
        tokenizer,
        input_ids,
        max_new_tokens,
    )
    print(compute_per_token_accuracy(model_output, input_ids, tokenizer))
