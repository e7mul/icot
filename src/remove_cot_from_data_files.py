import os

for name in ["test_bigbench", "train", "valid"]:
    file_path = f"data/4_by_4_mult/{name}.txt"

    with open(file_path, encoding="utf-8") as f:
        lines = [
            line.strip().split("||")
            for line in f.readlines()
            if (
                len(line.strip()) > 0
                and not line.strip().isspace()
                and len(line.strip().split("||")) == 2
            )
        ]
    new_lines = []
    for line in lines:
        new_line = line[0] + "%%% ####" + line[1].split("####")[-1] + "\n"
        new_lines.append(new_line)
        print(new_line)

    os.makedirs("data/4_by_4_mult_no_cot", exist_ok=True)

    file_path = f"data/4_by_4_mult_no_cot/{name}.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
