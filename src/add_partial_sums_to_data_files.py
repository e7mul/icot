import os

from data_utils import compute_partial_sums

path = "data/4_by_4_mult_no_cot"

for name in ["test_bigbench", "train", "valid"]:
    file_path = os.path.join(path, f"{name}.txt")

    with open(file_path, encoding="utf-8") as f:
        lines = [line.strip().split("%%%") for line in f.readlines()]
    new_lines = []

    for line in lines:
        operands = line[0]
        a, b = operands.split("*")
        a = a.replace(" ", "")
        b = b.replace(" ", "")
        partial_sums = compute_partial_sums(a, b)
        new_line = (
            line[0]
            + "%%% "
            + "& ".join(str(partial_sum) for partial_sum in partial_sums)
            + line[1]
            + "\n"
        )
        new_lines.append(new_line)

    os.makedirs("data/4_by_4_mult_no_cot_w_partial_sums", exist_ok=True)
    file_path = os.path.join("data/4_by_4_mult_no_cot_w_partial_sums", f"{name}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
