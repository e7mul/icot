from typing import List
import torch
from jaxtyping import Array, Float


def convert_str2ints_list(input: int) -> List[int]:
    return [int(character) for character in str(input)]


def get_sum_of_kth_antidiagonal(matrix: Float[Array, "n n"], k: int) -> List[int]:
    """
    Returns the sum of the elements on the k-th antidiagonal of a matrix.

    Since we multiply two integers with n elements, the result can be maximally
    of 2*n elements.
    """
    n = matrix.shape[0]
    assert (
        k <= 2 * n
    ), f"The result is maximmaly {2*n} long, and you asked for {k}-th element."

    return matrix.flip(dims=[0]).diagonal(offset=(n - 1) - k).sum().item()


def compute_partial_sums(a: str, b: str) -> List[int]:
    """
    Computes partial sums for digit multiplication according to Eq. 1 & 2
    from: arxiv.org/abs/2510.00184

    c_hut_k = s_k + r_{k-1}
    s_k = \sum_{i+j = k} a_i * b_j
    r_k = (s_k + r_{k-1}) // 10
    r_{-1} = 0

    """
    digit_length = len(a)

    a_as_tensor = torch.tensor(convert_str2ints_list(a))
    b_as_tensor = torch.tensor(convert_str2ints_list(b))

    cartesian_product = torch.outer(a_as_tensor, b_as_tensor)
    num_partial_sums = 2 * digit_length - 1
    partial_sums = []
    r = 0
    for i in range(num_partial_sums):
        s = get_sum_of_kth_antidiagonal(cartesian_product, k=i)
        c = s + r
        r = (c + r) // 10
        partial_sums.append(c)
    return partial_sums
