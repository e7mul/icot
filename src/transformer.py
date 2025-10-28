import torch, torch.nn as nn, torch.nn.functional as F
from typing import Optional, Any
from dataclasses import dataclass, field
import einops
from fancy_einsum import einsum
from jaxtyping import Float, Array, Integer
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedTokenizer
from src.ActivationCache import record_activations

# helper
ACT2FN = {
    "relu": F.relu,
    "gelu": F.gelu,
    "silu": F.silu,
    "swish": F.silu,
}


@dataclass
class Accuracies:
    total_correct_answers: int | None = None
    total_correct: int | None = None
    correct_ans_tokens: int | None = None
    total_ans_tokens: int | None = None
    total_tokens: int | None = None

    token_accuracy: float = field(init=False)

    def __post_init__(self):
        if self.total_tokens > 0:
            self.token_accuracy = self.total_correct / self.total_tokens
        else:
            self.token_accuracy = 0.0


@dataclass
class Losses:
    partial_sums_loss: torch.Tensor | None = None
    per_token_loss: torch.Tensor | None = None
    regress_output_loss: torch.Tensor | None = None

    token_loss: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.token_loss = self.per_token_loss.mean()


@dataclass
class Output:
    losses: Losses
    acc: Accuracies


@dataclass
class AttentionConfig:
    D: int = 768
    layer_idx: Optional[int] = None
    n_heads: int = 4
    causal: bool = True
    device: str = "cuda"


class HookPoint(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Attention(nn.Module):  # BSD -> BSD
    def __init__(self, layer_idx: int, config: AttentionConfig):
        super().__init__()
        self.layer_idx = layer_idx
        self.D = config.D
        self.n_heads = config.n_heads
        assert self.D % self.n_heads == 0
        self.head_dim = self.D // self.n_heads
        self.Wq = nn.Linear(self.D, self.D, bias=False)
        self.Wk = nn.Linear(self.D, self.D, bias=False)
        self.Wv = nn.Linear(self.D, self.D, bias=False)
        self.causal = config.causal
        self.Wo = nn.Linear(self.D, self.D, bias=False)
        self.device = config.device
        # Hook points
        self.hook_attn_pattern = HookPoint()
        self.hook_attn_output_per_head = HookPoint()

    def forward(
        self, x: torch.Tensor, kv_cache: Optional[Any] = None
    ) -> torch.Tensor:  # input is [B, S, D]
        B, S, D = x.shape

        # Make each QKV [B, S, D] --> [B, nh, S, hd]
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)  # all [B, S, D]

        Q = Q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)  # [B, nh, S, hd]
        K = K.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # update kv cache
        layer_idx = self.layer_idx
        if kv_cache is not None and layer_idx is not None:
            # its preallocated, just write to the memory of the cache using state of current_length
            kv_cache.update(layer_idx, K, V)
            K = kv_cache.keys[layer_idx][:, :, : kv_cache.current_length, :]
            V = kv_cache.values[layer_idx][:, :, : kv_cache.current_length, :]

        # [B, nh, S, hd] @ [B, nh, hd, S] -> [B, nh, S, S]
        scale = torch.sqrt(
            torch.tensor(self.head_dim, dtype=Q.dtype, device=self.device)
        )
        logits = (Q @ K.transpose(-2, -1)) / scale
        if self.causal:
            mask = torch.triu(torch.ones_like(logits), diagonal=1).bool()
            logits_masked = logits.masked_fill(mask, float("-inf"))
        else:
            logits_masked = logits

        A = F.softmax(logits_masked, dim=-1)  # [B, nh, S, S]
        # Hook attention pattern: [B, nh, S, S]
        A = self.hook_attn_pattern(A)

        preout = torch.einsum("bnxy,bnyd->bnxd", A, V)  # [B, nh, S, hd]

        W_O = einops.rearrange(
            self.Wo.weight.T,
            "(n_heads d_head) d_model -> n_heads d_head d_model",
            n_heads=self.n_heads,
        )
        attn_output_per_head = einsum(
            "batch n_heads seq d_head, n_heads d_head d_model -> batch n_heads seq d_model",
            preout,
            W_O,
        )  # [B, nh, S, D]
        # Reorder to [B, S, nh, D] and hook
        attn_output_per_head_seq = attn_output_per_head.transpose(1, 2)
        attn_output_per_head_seq = self.hook_attn_output_per_head(
            attn_output_per_head_seq
        )
        # Sum across heads -> [B, S, D]
        attn_out = attn_output_per_head_seq.sum(dim=2)
        return attn_out  # [B, S, D]


@dataclass
class MLPConfig:
    D: int
    hidden_multiplier: int = 4
    act: str = "gelu"
    device: Optional[torch.device] = None


# most important fact about MLP: it operates on each token independently, ie. D --> D
class MLP(nn.Module):
    def __init__(self, config: MLPConfig):
        super().__init__()
        self.D = config.D
        self.device = (
            config.device
            if config.device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.up_proj = nn.Linear(self.D, self.D * config.hidden_multiplier, bias=False)
        self.down_proj = nn.Linear(
            self.D * config.hidden_multiplier, self.D, bias=False
        )
        self.act = ACT2FN[config.act]
        # Hook point at MLP mid activation
        self.hook_mlp_mid = HookPoint()

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:  # BSD -> BSD automatically on last dim
        mid = self.act(self.up_proj(x))
        mid = self.hook_mlp_mid(mid)  # [B, S, D*mult]
        return self.down_proj(mid)


@dataclass
class LNConfig:
    D: int
    eps: float = 1e-9
    device: Optional[torch.device] = None


class LN(nn.Module):
    def __init__(self, config: LNConfig):
        super().__init__()
        self.D = config.D
        self.eps = config.eps
        self.device = (
            config.device
            if config.device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.mean_scale = nn.Parameter(torch.zeros(self.D))
        self.std_scale = nn.Parameter(torch.ones(self.D))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x is [B, S, D]
        mean = x.mean(dim=-1, keepdim=True)  # [B, S, 1]
        std = (x.var(dim=-1, keepdim=True) + self.eps) ** 0.5  # [B, S, 1]
        x_norm = (x - mean) / (std)
        return x_norm * self.std_scale + self.mean_scale


@dataclass
class TransformerLayerConfig:
    D: int = 768
    n_heads: int = 4
    device: Optional[torch.device] = None


class TransformerLayer(nn.Module):
    def __init__(self, layer_idx: int, config: TransformerLayerConfig):
        super().__init__()
        self.D = config.D
        self.layer_idx = layer_idx
        self.device = (
            config.device
            if config.device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        attn_config = AttentionConfig(
            D=self.D, n_heads=config.n_heads, device=self.device
        )
        mlp_config = MLPConfig(D=self.D, device=self.device)
        ln_config = LNConfig(D=self.D, device=self.device)

        self.attn = Attention(self.layer_idx, attn_config)
        self.mlp = MLP(mlp_config)
        self.ln1 = LN(ln_config)
        self.ln2 = LN(ln_config)
        # Residual stream hook points
        self.hook_resid_pre = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_resid_post = HookPoint()

    def forward(
        self, x: torch.Tensor, kv_cache: Optional[Any] = None, return_attn: bool = False
    ) -> torch.Tensor:  # x is BSD
        x = self.hook_resid_pre(x)
        ln1_out = self.ln1(x)
        attn_out = self.attn(ln1_out, kv_cache=kv_cache)
        x = x + attn_out
        x = self.hook_resid_mid(x)
        ln2_out = self.ln2(x)
        mlp_out = self.mlp(ln2_out)
        x = x + mlp_out
        x = self.hook_resid_post(x)
        if return_attn:
            return x, attn_out
        return x


@dataclass
class PositionalEmbeddingConfig:
    max_seq_len: int
    D: int
    device: Optional[torch.device] = None


class PositionalEmbedding(nn.Module):
    def __init__(self, config: PositionalEmbeddingConfig):
        super().__init__()
        self.device = (
            config.device
            if config.device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.pos_embedding = nn.Parameter(torch.randn(config.max_seq_len, config.D))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x is [B, S, D]
        B, S, D = x.shape
        return x + self.pos_embedding[:S]  # Broadcasting handles batch dimension


@dataclass
class EmbeddingLayerConfig:
    vocab_size: int
    D: int
    device: Optional[torch.device] = None


class EmbeddingLayer(nn.Module):
    def __init__(self, config: EmbeddingLayerConfig):
        super().__init__()
        self.device = (
            config.device
            if config.device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.embedding = nn.Parameter(torch.randn(config.vocab_size, config.D))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding[x]


@dataclass
class UnembeddingLayerConfig:
    vocab_size: int
    D: int
    device: Optional[torch.device] = None


class UnembeddingLayer(nn.Module):
    def __init__(self, config: UnembeddingLayerConfig):
        super().__init__()
        self.device = (
            config.device
            if config.device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.V = config.vocab_size
        self.unembedding = nn.Linear(config.D, self.V, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x is [B, S, D]
        return self.unembedding(x)


@dataclass
class TransformerConfig:
    hidden_dim: int = 768
    depth: int = 2
    n_heads: int = 4
    vocab_size: int = 50257
    max_seq_len: int = 128
    device: Optional[torch.device] = None


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.depth = config.depth
        self.hidden_dim = config.hidden_dim
        self.vocab_size = config.vocab_size
        self.tokenizer = tokenizer

        emb_config = EmbeddingLayerConfig(
            vocab_size=config.vocab_size, D=config.hidden_dim, device=config.device
        )
        pos_emb_config = PositionalEmbeddingConfig(
            max_seq_len=config.max_seq_len, D=config.hidden_dim, device=config.device
        )
        unemb_config = UnembeddingLayerConfig(
            vocab_size=config.vocab_size, D=config.hidden_dim, device=config.device
        )

        self.emb = EmbeddingLayer(emb_config)
        self.pos_emb = PositionalEmbedding(pos_emb_config)

        self.ln_final = LN(LNConfig(D=config.hidden_dim, device=config.device))
        self.unemb = UnembeddingLayer(unemb_config)

        layer_config = TransformerLayerConfig(
            D=config.hidden_dim, n_heads=config.n_heads, device=config.device
        )
        self.layers = nn.ModuleList(
            [TransformerLayer(idx, layer_config) for idx in range(config.depth)]
        )
        for i, layer in enumerate(self.layers):
            layer.attn.layer_idx = i

        self.partial_sum_predictors = nn.ModuleDict(
            modules={
                str(head_idx): nn.Linear(self.hidden_dim, 1)
                for head_idx in range(config.n_heads)
            }
        )

        self.output_mse_probe = nn.Linear(self.hidden_dim, 1)

        self.device = config.device

    def forward(
        self, x: torch.Tensor, kv_cache: Optional[Any] = None, return_attn: bool = False
    ) -> torch.Tensor:
        x = self.emb(x)
        if kv_cache is not None:
            # When decoding, only add positional embeddings for the new tokens.
            pos_offset = kv_cache.current_length
            pos_emb = self.pos_emb.pos_embedding[
                pos_offset : pos_offset + x.size(1)
            ].unsqueeze(0)
            x = x + pos_emb
        else:
            x = self.pos_emb(x)

        all_attn = []
        for _, layer in enumerate(self.layers):
            if return_attn:
                x, attn = layer(x, kv_cache=kv_cache, return_attn=True)
                all_attn.append(attn)
            else:
                x = layer(x, kv_cache=kv_cache)

        x = self.ln_final(x)
        logits = self.unemb(x)
        if return_attn:
            return logits, torch.stack(all_attn, dim=0), x
        return logits, None, x

    def compute_output(
        self,
        input_tokens: Float[Array, "batch_size seq_len"],
        module_for_attns: str,
    ) -> tuple[
        Float[Array, "batch_size seq_len vocab_size"],
        Float[Array, "batch_size seq_len n_heads h_dim//n_heads"],
        Float[Array, "batch_size seq_len h_dim"],
    ]:
        with record_activations(
            self, module_names=[module_for_attns], detach_activations=False
        ) as activs_cache:
            logits, _, hidden_states = self.forward(input_tokens)
        attentions = activs_cache[module_for_attns]
        return logits, attentions, hidden_states

    @staticmethod
    def cutoff_input_elements(labels: torch.Tensor) -> torch.Tensor:
        """
        Given a tensor of shape [batch_size, seq_len], where each row starts
        with zero or more -100 values followed by target values, return a tensor
        of shape [batch_size, target_len] containing only the target values
        (i.e., excluding the leading -100s).

        Assumes all rows have the same number of leading -100s.
        """
        # Find number of leading -100s (assume same for every row)
        mask = labels[0] == -100
        num_leading = mask.long().sum().item()
        return labels[:, num_leading:]

    def tokens2ints(
        self, labels: Integer[Array, "batch_size ..."]
    ) -> Float[Array, "batch_size ..."]:
        tokens = self.tokenizer.batch_decode(labels)
        result = []
        for sample in tokens:
            cleaned = sample.replace(" ", "")
            try:
                sample_tensor = torch.tensor([float(digit) for digit in cleaned])
            except ValueError as e:
                raise ValueError(
                    f"Non-numeric character in decoded token: {cleaned}"
                ) from e
            result.append(sample_tensor)
        return torch.stack(result).to(self.device)

    def compute_mse_loss(
        self,
        hidden_states: Float[Array, "batch_size seq_len h_dim"],
        labels: Integer[Array, "batch_size seq_len"],
    ) -> Float[Array, ""]:
        labels = self.cutoff_input_elements(labels)
        target_len = labels.shape[-1]
        hidden_states = hidden_states[:, -target_len:, :]
        labels = self.tokens2ints(labels)
        probe_output = self.output_mse_probe(hidden_states).squeeze(-1)
        assert (
            probe_output.shape == labels.shape
        ), f"Inconsistent shapes when computing MSE Loss. Output probe {probe_output.shape}, Targets: {labels.shape}"
        return torch.nn.functional.mse_loss(probe_output, labels)

    def compute_loss(
        self,
        input_tokens: Integer[Array, "batch_size seq_len"],
        target_tokens: Integer[Array, "batch_size seq_len"],
        partial_sums: Float[Array, "batch_size num_partials"],
        separator_position: int,
        module_for_attns: str = "layers.1.attn.hook_attn_output_per_head",
    ):

        logits, attns, hidden_states = self.compute_output(
            input_tokens, module_for_attns
        )

        accuracies = compute_accuracies(logits, target_tokens, separator_position)

        losses = Losses(
            partial_sums_loss=self.compute_partial_sums_loss(
                attns, partial_sums, sep_pos=separator_position
            ),
            per_token_loss=compute_next_token_loss(logits, target_tokens),
            regress_output_loss=self.compute_mse_loss(hidden_states, target_tokens),
        )

        out = Output(losses=losses, acc=accuracies)
        return out

    def compute_partial_sums_loss(
        self,
        attention_maps: Float[Array, "batch_size seq_len n_heads h_dim/n_heads"],
        partial_sums: Float[Array, "batch_size num_partial_sums"],
        sep_pos: int,
    ) -> torch.Tensor:
        attention_maps = attention_maps[:, sep_pos + 1 : -1, ...]
        total_loss = torch.tensor(
            0.0, device=attention_maps.device, dtype=attention_maps.dtype
        )
        for head_idx, linear_probe in self.partial_sum_predictors.items():
            output = linear_probe(attention_maps[..., int(head_idx), :]).squeeze(2)
            assert (
                output.shape == partial_sums.shape
            ), f"Inconsistent shapes when computing MSE Loss. Output {output.shape}, Partial sums: {partial_sums.shape}"
            total_loss += torch.nn.functional.mse_loss(output, partial_sums)
        return total_loss


def compute_accuracies(
    logits: Float[Array, "batch_size seq_len vocab_size"],
    labels: Float[Array, "batch_size seq_len"],
    separator_position: int,
) -> Accuracies:

    # the last input token is <|endoftext|> thus, we don't care what models predicts for it
    ans_preds = logits[..., separator_position:-1, :].argmax(-1)
    # here we add +1 as labels[..., sep_position] is the second-to-last EoS token
    ans_labels = labels[..., separator_position + 1 :]
    correct_per_row = (ans_preds == ans_labels).sum(-1)

    labels_pred = logits.argmax(-1)
    mask = labels[..., 1:].ge(0)
    correct_toks = ((labels_pred[..., :-1] == labels[..., 1:]) * mask).sum()

    acc = Accuracies(
        correct_ans_tokens=(ans_preds == ans_labels).sum().item(),
        total_ans_tokens=(ans_labels != -100).sum().item(),
        total_correct_answers=(correct_per_row == ans_labels.shape[-1]).sum().item(),
        total_correct=correct_toks.item(),
        total_tokens=mask.sum().item(),
    )
    return acc


def compute_next_token_loss(
    logits: Float[Array, "batch_size seq_len vocab_size"],
    target_tokens: Float[Array, "batch_size seq_len"],
) -> torch.Tensor:
    # when comparing target_tokens with predictions we skip the first target_token as it is already sent as the input to the model
    # and we don't care about the last prediction of the model as it predict the next token after <|endoftext|>
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = target_tokens[..., 1:].contiguous()
    loss_fct = CrossEntropyLoss(reduction="none")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(shift_labels.shape).mean(dim=0)
    return loss
