"""Auto-AVSR attention decoder and joint CTC/attention beam search.

Mirrors ESPnet's transformer decoder (pre-norm, ReLU feed-forward, absolute
positional encoding) and its joint decoding rule: score = (1 - w) * attention
+ w * CTC prefix score, with CTC prefix scoring restricted to the attention
decoder's top 1.5 * beam candidates. The published 19.1% configuration also
adds an external subword RNN language model (weight 0.3), not ported here.
"""
import math
from pathlib import Path

import torch
from torch import nn

from VisualPhoneme.autoavsr_encoder import AutoAvsrCtc, FeedForward, load_autoavsr_ctc


class MultiHeadedAttention(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads, self.head_width = heads, width // heads
        self.linear_q = nn.Linear(width, width)
        self.linear_k = nn.Linear(width, width)
        self.linear_v = nn.Linear(width, width)
        self.linear_out = nn.Linear(width, width)

    def forward(self, query: torch.Tensor, memory: torch.Tensor,
                allowed: torch.Tensor) -> torch.Tensor:
        """allowed broadcasts to [batch,query_time,key_time]; True may attend."""
        split = lambda x: x.view(x.shape[0], -1, self.heads, self.head_width).transpose(1, 2)
        q, k, v = split(self.linear_q(query)), split(self.linear_k(memory)), split(self.linear_v(memory))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_width)
        blocked = ~allowed[:, None]
        weights = scores.masked_fill(blocked, torch.finfo(scores.dtype).min).softmax(-1)
        context = (weights.masked_fill(blocked, 0.0) @ v).transpose(1, 2)
        return self.linear_out(context.reshape(query.shape[0], -1, self.heads * self.head_width))


class PositionalEncoding(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.width, self.scale = width, math.sqrt(width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        position = torch.arange(inputs.shape[1], device=inputs.device, dtype=torch.float32)[:, None]
        frequency = torch.exp(torch.arange(0, self.width, 2, device=inputs.device,
                                           dtype=torch.float32)
                              * -(math.log(10000.0) / self.width))
        encoding = torch.zeros(inputs.shape[1], self.width, device=inputs.device)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency)
        return inputs * self.scale + encoding.to(inputs.dtype)


class DecoderLayer(nn.Module):
    def __init__(self, width: int, heads: int, hidden: int):
        super().__init__()
        self.self_attn = MultiHeadedAttention(width, heads)
        self.src_attn = MultiHeadedAttention(width, heads)
        self.feed_forward = FeedForward(width, hidden, 0.0)
        self.norm1 = nn.LayerNorm(width, eps=1e-12)
        self.norm2 = nn.LayerNorm(width, eps=1e-12)
        self.norm3 = nn.LayerNorm(width, eps=1e-12)

    def forward(self, tokens: torch.Tensor, causal: torch.Tensor, memory: torch.Tensor,
                memory_valid: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(tokens)
        x = tokens + self.self_attn(normed, normed, causal)
        x = x + self.src_attn(self.norm2(x), memory, memory_valid)
        return x + self.feed_forward(self.norm3(x))


class AutoAvsrDecoder(nn.Module):
    def __init__(self, vocabulary: int = 5049, width: int = 768, heads: int = 12,
                 hidden: int = 3072, layers: int = 6):
        super().__init__()
        self.embed = nn.Sequential(nn.Embedding(vocabulary, width), PositionalEncoding(width))
        self.decoders = nn.ModuleList(DecoderLayer(width, heads, hidden) for _ in range(layers))
        self.after_norm = nn.LayerNorm(width, eps=1e-12)
        self.output_layer = nn.Linear(width, vocabulary)

    def next_token_log_probabilities(self, prefixes: torch.Tensor, memory: torch.Tensor,
                                     memory_valid: torch.Tensor) -> torch.Tensor:
        """prefixes [batch,length] starting with <sos>; returns [batch,vocabulary]."""
        length = prefixes.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool, device=prefixes.device).tril()[None]
        x = self.embed(prefixes)
        for layer in self.decoders:
            x = layer(x, causal, memory, memory_valid[:, None, :])
        return self.output_layer(self.after_norm(x[:, -1])).log_softmax(-1)


def load_autoavsr_joint(checkpoint: Path) -> tuple[AutoAvsrCtc, AutoAvsrDecoder]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    decoder = AutoAvsrDecoder(state["decoder.output_layer.weight"].shape[0])
    decoder.load_state_dict({key.removeprefix("decoder."): value for key, value in state.items()
                             if key.startswith("decoder.")}, strict=True)
    return load_autoavsr_ctc(checkpoint), decoder


def _ctc_extend(log_ctc: torch.Tensor, states: torch.Tensor, last: torch.Tensor,
                candidates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """CTC prefix scores of every hypothesis extended by each of its candidates.

    log_ctc [T,V]; states [H,T,2] holding log r^n, r^b of each prefix (Watanabe
    et al., 2017); last [H] final token or -1 for the empty prefix; candidates
    [H,K]. Returns ([H,K,T,2] extended states, [H,K] prefix log probabilities).
    """
    steps = log_ctc.shape[0]
    x = log_ctc[:, candidates]                                      # [T,H,K]
    r_b_prefix = states[:, :, 1].T[:, :, None]                      # [T,H,1]
    total = torch.logaddexp(states[:, :, 0], states[:, :, 1]).T[:, :, None]
    # A repeated token needs an intervening blank, so only r^b of the prefix counts.
    phi = torch.where((candidates == last[:, None])[None], r_b_prefix, total)
    empty = (last < 0)[:, None]
    r_n = torch.full_like(x, -math.inf)
    r_b = torch.full_like(x, -math.inf)
    r_n[0] = torch.where(empty, x[0], -math.inf)
    for t in range(1, steps):
        r_n[t] = torch.logaddexp(r_n[t - 1], phi[t - 1]) + x[t]
        r_b[t] = torch.logaddexp(r_n[t - 1], r_b[t - 1]) + log_ctc[t, 0]
    prefix = torch.logsumexp(torch.cat((r_n[:1], phi[:-1] + x[1:])), dim=0)
    return torch.stack((r_n, r_b), dim=-1).permute(1, 2, 0, 3), prefix


@torch.inference_mode()
def joint_beam_search(log_ctc: torch.Tensor, memory: torch.Tensor, decoder: AutoAvsrDecoder,
                      beam: int = 40, ctc_weight: float = 0.1) -> list[int]:
    """Decode one utterance. log_ctc [time,vocab]; memory [time,width]. Returns unit ids."""
    steps, vocabulary = log_ctc.shape
    eos = sos = vocabulary - 1
    device = log_ctc.device
    pre_beam = min(vocabulary - 2, int(1.5 * beam))
    prefixes = torch.full((1, 1), sos, dtype=torch.long, device=device)
    scores = torch.zeros(1, device=device)
    states = torch.stack((torch.full((steps,), -math.inf, device=device),
                          log_ctc[:, 0].cumsum(0)), dim=1)[None]     # [H,T,2]
    ctc_prefix = torch.zeros(1, device=device)
    last = torch.full((1,), -1, dtype=torch.long, device=device)
    ended: list[tuple[float, list[int]]] = []
    memory_valid = torch.ones(1, steps, dtype=torch.bool, device=device)
    for _ in range(steps):
        count = len(prefixes)
        attention = decoder.next_token_log_probabilities(
            prefixes, memory[None].expand(count, -1, -1), memory_valid.expand(count, -1)
        )
        attention[:, 0] = -math.inf          # blank is a CTC symbol, never an output token
        eos_attention = attention[:, eos].clone()
        attention[:, eos] = -math.inf
        candidates = attention.topk(pre_beam, dim=1).indices                       # [H,K]
        extended, prefix = _ctc_extend(log_ctc, states, last, candidates)
        token_scores = (scores[:, None]
                        + (1 - ctc_weight) * attention.gather(1, candidates)
                        + ctc_weight * (prefix - ctc_prefix[:, None]))
        final_ctc = torch.logaddexp(states[:, -1, 0], states[:, -1, 1])
        eos_scores = scores + (1 - ctc_weight) * eos_attention + ctc_weight * (final_ctc - ctc_prefix)
        # ESPnet only considers <eos> when it is inside the attention pre-beam.
        eos_rank = (attention > eos_attention[:, None]).sum(1)
        eos_scores = torch.where(eos_rank < pre_beam, eos_scores, -math.inf)
        flat = torch.cat((token_scores.flatten(), eos_scores))
        best = flat.topk(min(beam, int(torch.isfinite(flat).sum()))).indices
        token_count = token_scores.numel()
        keep_rows, keep_columns = [], []
        for index, score in zip(best.tolist(), flat[best].tolist()):
            if index >= token_count:
                ended.append((score, prefixes[index - token_count, 1:].tolist()))
            else:
                keep_rows.append(index // pre_beam)
                keep_columns.append(index % pre_beam)
        if not keep_rows:
            break
        rows = torch.tensor(keep_rows, device=device)
        columns = torch.tensor(keep_columns, device=device)
        tokens = candidates[rows, columns]
        prefixes = torch.cat((prefixes[rows], tokens[:, None]), dim=1)
        scores = token_scores[rows, columns]
        states = extended[rows, columns]
        ctc_prefix = prefix[rows, columns]
        last = tokens
        # Scores only decrease as hypotheses extend, so a better ended one is final.
        if ended and scores.max().item() <= max(score for score, _ in ended):
            break
    if not ended:
        best_row = int(scores.argmax())
        return prefixes[best_row, 1:].tolist()
    return max(ended, key=lambda item: item[0])[1]
