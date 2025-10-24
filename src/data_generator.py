import os
import argparse
import numpy as np

from src.partial_sums_utils import compute_partial_sums


def int2str(num: int, num_digits: int) -> str:
    """
    Converts an integer to a string with:
        1. each digit separated by a whitespace
        2. padded with zeros to have always num_digits
        3. reversed order to have causual-friendly structure for Transformers
    """
    return " ".join(str(num)[::-1].rjust(num_digits, "0"))


def check_for_copy(line: str, all_lines_set: set[str]):
    """
    Checks if the line is not already within the all_lines_set (highly efficient, use a set!).
    """
    return line not in all_lines_set


def generate_data(
    digits_in_operands: int,
    num_samples: int,
    add_partial_sums: bool,
    previous_lines: list[str],
) -> list[str]:
    maximum_digit = int("9" * digits_in_operands)
    all_lines = []
    while len(all_lines) < num_samples:
        a, b = np.random.randint(1, maximum_digit, size=(2,))
        result = a * b
        a_str = int2str(a, digits_in_operands)
        b_str = int2str(b, digits_in_operands)
        res_str = int2str(result, 2 * digits_in_operands)

        line = f"{a_str} * {b_str}%%% #### {res_str}"
        if add_partial_sums:
            partial_a = a_str.replace(" ", "")[::-1]
            partial_b = b_str.replace(" ", "")[::-1]
            partial_sums = compute_partial_sums(partial_a, partial_b)
            line.replace("\n", "")
            line += "&&&" + "& ".join(str(partial_sum) for partial_sum in partial_sums)
        if check_for_copy(line, all_lines):
            if previous_lines:
                if check_for_copy(line, previous_lines):
                    all_lines.append(line + "\n")
            else:
                all_lines.append(line + "\n")
    return all_lines


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_digits",
        type=int,
        required=True,
        help="Number of digits in each operand (e.g., for 4-digit multiplication, use 4).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        required=True,
        help="Number of multiplication samples to generate.",
    )
    parser.add_argument(
        "--fname",
        type=str,
        required=True,
        help="Base filename for the generated dataset (without extension).",
    )
    parser.add_argument(
        "--generate_partial_sums",
        action="store_true",
        required=False,
        help="Flag to generate partial sums in the dataset",
    )
    parser.add_argument(
        "--previous_datasets",
        type=str,
        required=False,
        nargs="+",
        help="Space separated list of file names to make sure there is no overlap between train, val and test set examples",
    )
    args = parser.parse_args()

    num_digits = args.num_digits
    num_samples = args.num_samples

    path = os.path.join("data", f"{num_digits}_by_{num_digits}")
    os.makedirs(path, exist_ok=True)

    previous_examples = []
    if args.previous_datasets:
        for fname in args.previous_datasets:
            with open(os.path.join(path, fname + ".txt"), "r") as f:
                previous_examples += f.readlines()

    all_lines = generate_data(
        num_digits, num_samples, args.generate_partial_sums, previous_examples
    )

    with open(os.path.join(path, f"{args.fname}.txt"), "w") as f:
        f.writelines(all_lines)
