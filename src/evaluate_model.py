import argparse
import torch

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from jaxtyping import Float, Array

from src.model_utils import load_c_hat_model
from src.train import get_sep_position
from src.data_utils import PSDataset


def greedy_search(
    model: torch.nn.Module,
    input_ids: Float[Array, "n_samples seq_len"],
) -> Float[Array, "n_samples 1"]:
    output = model.forward(input_ids)
    logits = output.logits
    selected_tokens = logits[:, -1, :].argmax(dim=-1).view(-1, 1)
    return selected_tokens


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    input_ids: Float[Array, "n_samples seq_len"],
    max_new_tokens: int,
) -> Float[Array, "n_samples seq_len"]:

    model.eval()
    sep_position = get_sep_position(tokenizer, input_ids)
    generated_ids = input_ids[:, : sep_position + 1]
    for i in range(max_new_tokens):
        selected_tokens = greedy_search(model, generated_ids)
        generated_ids = torch.cat((generated_ids, selected_tokens), dim=1)
    return generated_ids


def get_data(
    data_path: str, tokenizer: PreTrainedTokenizerBase
) -> Float[Array, "n_samples seq_len"]:
    dataset = PSDataset(tokenizer, data_path)
    return torch.tensor(dataset.examples_all, dtype=torch.long)


def compute_per_token_accuracy(
    output: Float[Array, "n_samples seq_len"],
    target: Float[Array, "n_samples seq_len"],
    tokenizer: PreTrainedTokenizerBase,
) -> Float[Array, "output_len"]:
    sep_position = get_sep_position(tokenizer, input_ids)
    output = output[:, sep_position + 1 :]
    target = target[:, sep_position + 1 :]
    return (output == target).float().mean(dim=0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="results/4_by_4_mult/gpt2_20251020_131030/checkpoints/checkpoint_44.pt",
    )  # required=True)
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/5_by_5_no_cot/valid.txt",
    )  # required=True)
    parser.add_argument("--gpu_ord", type=int, required=False, default=0)
    args = parser.parse_args()

    max_new_tokens = 2 * int(args.data_path.split("/")[1].split("_")[0])
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
        gen_func="greedy_search",
    )

    print(compute_per_token_accuracy(input_ids, model_output, tokenizer))
