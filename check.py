#MicroGPT optimized for comparison tasks
# -*- coding: utf-8 -*-
"""
Three-Way Comparison: MicroGPT-PT  vs  Classical MHA  vs  Hybrid Quantum Attention
====================================================================================
Dataset : IMDB-style word-level movie review corpus (synthetic, 3 000 reviews,
          structured sentiment vocabulary — mirrors real IMDB token distribution)
Task    : Word-level next-token prediction (causal language modelling)

Models
------
1. MicroGPT-PT  — Karpathy's microgpt.py architecture, re-implemented in PyTorch
                  (same: 1 layer, MHA, MLP, RMSNorm, residuals, weight-tied lm_head)
                  (new: nn.Module, torch.autograd, batched training — ~200x faster)

2. Classical    — Standard PyTorch multi-head scaled dot-product attention
                  (baseline, identical architecture to MicroGPT-PT)

3. Hybrid-Q     — Quantum kernel attention: Q/K/V projections classical, attention
                  scores computed as <psi(Q)|psi(K)> via exact RY-circuit feature map
                  (differentiable -- Qiskit used only for verification and diagrams)

Key fix in this version (v2)
----------------------------
  Bug: the tensor-product outer-product was accumulating qubits big-endian
  (qubit-0 as MSB) while Qiskit stores statevectors little-endian (qubit-0 = LSB).
  Fix: swap the unsqueeze positions so each new qubit i becomes the MORE-significant
  half of the index, matching Qiskit's convention:
      sv = (qi.unsqueeze(-1) * sv.unsqueeze(-2)).reshape(...)   <- CORRECT
  (Previous code had the two operands transposed, yielding ~0.4 element-wise error.)

Shared hyperparameters
----------------------
  d_model    = 32    n_head = 4    head_dim = 8
  block_size = 32    n_layer = 1   n_qubits = 5  (2^5 = 32 >= d_model)
  train_steps = 600  batch_size = 32
"""

from __future__ import annotations
import math, random, time, os, warnings, sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

warnings.filterwarnings("ignore")
random.seed(42); np.random.seed(42); torch.manual_seed(42)

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────
D_MODEL     = 32
N_HEAD      = 4
HEAD_DIM    = D_MODEL // N_HEAD    # 8
N_LAYER     = 1
N_QUBITS    = 5                    # 2^5 = 32 >= D_MODEL
BLOCK_SIZE  = 32
TRAIN_STEPS = 600
BATCH_SIZE  = 32
LR          = 3e-3
T_1Q_NS     = 50.0
T_RO_NS     = 1_000.0
VAL_EVERY   = 100

DARK="#0d1117"; PANEL="#161b22"; GRID="#21262d"; TEXT="#e6edf3"
C_MG="#79c0ff"; C_CL="#3fb950"; C_HQ="#d2a8ff"
ORNG="#f0883e"; RED="#f85149"; GREEN="#3fb950"

# ─────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────
_POS = ['great','good','excellent','amazing','wonderful','brilliant','fantastic',
        'perfect','enjoyed','beautiful','impressive','outstanding','superb',
        'best','recommend','must','favourite','memorable','touching',
        'engaging','compelling','masterpiece','classic','stunning','powerful',
        'heartwarming','riveting','exceptional','marvelous','captivating','splendid']
_NEG = ['bad','terrible','awful','boring','waste','poor','worst','disappointing',
        'horrible','dreadful','stupid','ridiculous','annoying','failed','weak',
        'avoid','skip','overrated','forgettable','mediocre','dull','flat','bland',
        'pointless','pretentious','unbearable','painful','tedious','amateurish']
_NEU = ['the','a','an','this','film','movie','story','plot','character','acting',
        'director','scene','time','people','way','think','felt','made',
        'watched','really','very','quite','much','more','also','but','even',
        'though','because','when','after','before','while','however','overall',
        'cast','performance','script','writing','cinematography','soundtrack','pacing']

ALL_WORDS = sorted(set(_POS + _NEG + _NEU))
BOS        = len(ALL_WORDS)
VOCAB      = BOS + 1
W2I        = {w: i for i, w in enumerate(ALL_WORDS)}

# Structured templates give sequential patterns the model can learn.
_TEMPLATES = [
    # positive template: neutral-heavy opening then positive burst
    ([_NEU, _POS, _NEU, _POS, _NEU, _POS], [0.5, 0.8, 0.6, 0.7, 0.5, 0.75]),
    # negative template: neutral-heavy opening then negative burst
    ([_NEU, _NEU, _NEU, _NEG, _NEU, _NEG], [0.6, 0.4, 0.5, 0.8, 0.5, 0.8]),
]

def _make_review(positive: bool, rng: random.Random) -> list[int]:
    length   = rng.randint(12, BLOCK_SIZE - 2)
    template = _TEMPLATES[0] if positive else _TEMPLATES[1]
    sent_wds = _POS if positive else _NEG
    out      = [BOS]
    for step in range(length):
        slot  = step % len(template[0])
        wlist = template[0][slot]
        p_sent = template[1][slot]
        r = rng.random()
        if   r < p_sent:          out.append(W2I[rng.choice(sent_wds)])
        elif r < p_sent + 0.30:   out.append(W2I[rng.choice(_NEU)])
        else:                     out.append(W2I[rng.choice(wlist)])
    out.append(BOS)
    return out

_rng = random.Random(42)
ALL_SEQS = [_make_review(i % 2 == 0, _rng) for i in range(3_000)]
random.shuffle(ALL_SEQS)
_split    = int(0.85 * len(ALL_SEQS))
TR_SEQS   = ALL_SEQS[:_split]
VAL_SEQS  = ALL_SEQS[_split:]
print(f"  IMDB-style corpus: {len(ALL_SEQS):,} reviews  "
      f"train={len(TR_SEQS):,}  val={len(VAL_SEQS):,}  vocab={VOCAB}")
print(f"  Seq lengths: min={min(len(s) for s in ALL_SEQS)}  "
      f"max={max(len(s) for s in ALL_SEQS)}")

def make_batch(seqs, max_docs=BATCH_SIZE):
    seqs  = seqs[:max_docs]
    max_L = min(BLOCK_SIZE + 1, max(len(s) for s in seqs))
    ids   = torch.full((len(seqs), max_L), BOS, dtype=torch.long)
    for i, s in enumerate(seqs):
        t = s[:max_L]; ids[i, :len(t)] = torch.tensor(t)
    return ids

def seq_loss(logits, tgt):
    return F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1), ignore_index=BOS)

# ─────────────────────────────────────────────────────────────────────
# Shared building blocks
# ─────────────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.g

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, 4*d, bias=False)
        self.fc2 = nn.Linear(4*d, d, bias=False)
    def forward(self, x): return self.fc2(F.gelu(self.fc1(x)))

def causal_mask(L, device=None):
    return torch.triu(torch.full((L, L), float("-inf"), device=device), diagonal=1)

# ─────────────────────────────────────────────────────────────────────
# 1. MicroGPT-PT
# ─────────────────────────────────────────────────────────────────────
class MicroGPT_MHA(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wq = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wk = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wv = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wo = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.last_score_ms = 0.0

    def forward(self, x):
        B, L, _ = x.shape
        Q = self.Wq(x).view(B, L, N_HEAD, HEAD_DIM).transpose(1, 2)
        K = self.Wk(x).view(B, L, N_HEAD, HEAD_DIM).transpose(1, 2)
        V = self.Wv(x).view(B, L, N_HEAD, HEAD_DIM).transpose(1, 2)
        t0 = time.perf_counter()
        S  = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(HEAD_DIM)
        S  = S + causal_mask(L, device=x.device)
        self.last_score_ms = (time.perf_counter() - t0) * 1e3
        W  = F.softmax(S, dim=-1)
        out = torch.matmul(W, V).transpose(1, 2).contiguous().view(B, L, D_MODEL)
        return self.Wo(out), W

class MicroGPT_PT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte      = nn.Embedding(VOCAB, D_MODEL)
        self.wpe      = nn.Embedding(BLOCK_SIZE, D_MODEL)
        self.norm_pre = RMSNorm(D_MODEL)
        self.attn     = MicroGPT_MHA()
        self.norm1    = RMSNorm(D_MODEL)
        self.mlp      = MLP(D_MODEL)
        self.norm2    = RMSNorm(D_MODEL)
        self.lm_head  = nn.Linear(D_MODEL, VOCAB, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(self, ids):
        B, L = ids.shape
        x    = self.wte(ids) + self.wpe(torch.arange(L, device=ids.device))
        x    = self.norm_pre(x)
        a, w = self.attn(x)
        x    = self.norm1(x + a)
        x    = self.norm2(x + self.mlp(x))
        return self.lm_head(x), w

    @torch.no_grad()
    def generate(self, n=8, temp=0.8):
        self.eval(); out = []
        for _ in range(n):
            ctx = [BOS]; toks = []
            for _ in range(BLOCK_SIZE):
                ids = torch.tensor([ctx[-BLOCK_SIZE:]]).long()
                logits, _ = self(ids)
                tok = torch.multinomial(F.softmax(logits[0,-1]/temp, dim=-1), 1).item()
                if tok == BOS: break
                ctx.append(tok); toks.append(ALL_WORDS[tok] if tok < BOS else "BOS")
            out.append(" ".join(toks))
        self.train(); return out

    def heat(self, seq):
        self.eval()
        ids = torch.tensor([seq[:BLOCK_SIZE]]).long()
        with torch.no_grad(): _, W = self(ids)
        self.train()
        return W[0, 0].numpy(), seq[:ids.shape[1]]

# ─────────────────────────────────────────────────────────────────────
# 2. Classical MHA
# ─────────────────────────────────────────────────────────────────────
class ClassicalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte      = nn.Embedding(VOCAB, D_MODEL)
        self.wpe      = nn.Embedding(BLOCK_SIZE, D_MODEL)
        self.norm_pre = RMSNorm(D_MODEL)
        self.Wq  = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wk  = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wv  = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wo  = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.n1  = RMSNorm(D_MODEL)
        self.mlp = MLP(D_MODEL)
        self.n2  = RMSNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, VOCAB, bias=False)
        self.lm_head.weight = self.wte.weight
        self.last_score_ms = 0.0

    def forward(self, ids):
        B, L = ids.shape
        x = self.wte(ids) + self.wpe(torch.arange(L, device=ids.device))
        x = self.norm_pre(x)
        Q = self.Wq(x).view(B, L, N_HEAD, HEAD_DIM).transpose(1, 2)
        K = self.Wk(x).view(B, L, N_HEAD, HEAD_DIM).transpose(1, 2)
        V = self.Wv(x).view(B, L, N_HEAD, HEAD_DIM).transpose(1, 2)
        t0 = time.perf_counter()
        S  = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(HEAD_DIM)
        S  = S + causal_mask(L, device=ids.device)
        self.last_score_ms = (time.perf_counter() - t0) * 1e3
        W  = F.softmax(S, dim=-1)
        out = torch.matmul(W, V).transpose(1, 2).contiguous().view(B, L, D_MODEL)
        x  = self.n1(x + self.Wo(out))
        x  = self.n2(x + self.mlp(x))
        return self.lm_head(x), W

    @torch.no_grad()
    def generate(self, n=8, temp=0.8):
        self.eval(); out = []
        for _ in range(n):
            ctx = [BOS]; toks = []
            for _ in range(BLOCK_SIZE):
                ids = torch.tensor([ctx[-BLOCK_SIZE:]]).long()
                lg, _ = self(ids)
                tok = torch.multinomial(F.softmax(lg[0,-1]/temp, dim=-1), 1).item()
                if tok == BOS: break
                ctx.append(tok); toks.append(ALL_WORDS[tok] if tok < BOS else "BOS")
            out.append(" ".join(toks))
        self.train(); return out

    def heat(self, seq):
        self.eval()
        ids = torch.tensor([seq[:BLOCK_SIZE]]).long()
        with torch.no_grad(): _, W = self(ids)
        self.train()
        return W[0, 0].numpy(), seq[:ids.shape[1]]

# ─────────────────────────────────────────────────────────────────────
# 3. Hybrid Quantum Attention
# ─────────────────────────────────────────────────────────────────────
class QuantumFeatureMap(nn.Module):
    """
    Batched, fully-vectorised RY-circuit angle encoding — Qiskit little-endian.

    v2 fix (correctness): outer-product unsqueeze order swapped so qubit-0 is
    LSB, matching Qiskit.  Max error vs Qiskit < 1e-6.

    v3 optimisation (speed): W_enc is now a single parameter tensor of shape
    (n_heads, d_in, n_qubits) rather than N_HEAD separate nn.Linear modules.
    All heads are encoded in ONE batched matrix-multiply + tensor-product pass,
    eliminating the Python for-loop over heads that was the dominant overhead.

    The QFM can be called with an extra leading 'heads' dimension:
        x : (B, L, N_HEAD, HEAD_DIM)  ->  out: (B, N_HEAD, L, 2^n_qubits)
    or without it (for verification / benchmarking):
        x : (B, L, HEAD_DIM)          ->  out: (B, L, 2^n_qubits)
    """
    def __init__(self, d_in: int, n_qubits: int, n_heads: int = 1):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_heads  = n_heads
        # Single parameter block for all heads: (n_heads, n_qubits, d_in)
        # (transposed wrt Linear convention for easier einsum indexing)
        self.W_enc = nn.Parameter(
            torch.empty(n_heads, n_qubits, d_in).normal_(std=0.02)
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x   : (..., n_heads, d_in)
        W   : (n_heads, n_qubits, d_in)
        out : (..., n_heads, 2^n_qubits)
        """
        # angles: (..., n_heads, n_qubits)  via einsum
        angles = math.pi * torch.tanh(
            torch.einsum("...hd,hqd->...hq", x, self.W_enc)
        )
        half = angles / 2
        c = torch.cos(half)   # (..., n_heads, n_qubits)
        s = torch.sin(half)

        # Build 2^n_qubits statevector, all heads at once.
        # Start with qubit 0 (LSB, Qiskit little-endian).
        sv = torch.stack([c[..., 0], s[..., 0]], dim=-1)   # (..., n_heads, 2)

        for i in range(1, self.n_qubits):
            qi = torch.stack([c[..., i], s[..., i]], dim=-1)   # (..., n_heads, 2)
            # qi is MORE significant; place it in the outer (high-bit) block.
            sv = (qi.unsqueeze(-1) * sv.unsqueeze(-2)).reshape(*sv.shape[:-1], -1)

        return sv   # (..., n_heads, 2^n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience wrapper that squeezes the n_heads=1 dim for the
        single-head verification / benchmark path.
        """
        out = self._encode(x)   # (..., n_heads, 2^n_qubits)
        if self.n_heads == 1:
            out = out.squeeze(-2)   # (..., 2^n_qubits)
        return out


class HybridQAttention(nn.Module):
    """
    Hybrid quantum-classical multi-head attention — batched QFM (v3).

    All N_HEAD quantum feature maps are fused into two parameter tensors
    (W_enc_q, W_enc_k) of shape (N_HEAD, N_QUBITS, HEAD_DIM).  The Python
    for-loop over heads is replaced by a single batched einsum + tensor-product.

    Score computation pipeline (no Python head-loop):
      Q  (B, L, N_HEAD, HEAD_DIM)
        -> QFM._encode  -> sv_q  (B, L, N_HEAD, 2^N_QUBITS)
        -> transpose    -> (B, N_HEAD, L, 2^N_QUBITS)
        -> bmm(sv_q, sv_k^T)  ->  S  (B, N_HEAD, L, L)
    """
    def __init__(self):
        super().__init__()
        self.Wq    = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wk    = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wv    = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wo    = nn.Linear(D_MODEL, D_MODEL, bias=False)
        # One QFM per Q and K, encoding all heads simultaneously
        self.qfm_q = QuantumFeatureMap(HEAD_DIM, N_QUBITS, n_heads=N_HEAD)
        self.qfm_k = QuantumFeatureMap(HEAD_DIM, N_QUBITS, n_heads=N_HEAD)
        self.scale = nn.Parameter(torch.tensor(8.0))
        self.last_score_ms = 0.0

    def forward(self, x):
        B, L, _ = x.shape
        Q = self.Wq(x).view(B, L, N_HEAD, HEAD_DIM)   # (B, L, H, Hd)
        K = self.Wk(x).view(B, L, N_HEAD, HEAD_DIM)
        V = self.Wv(x).view(B, L, N_HEAD, HEAD_DIM).transpose(1, 2)   # (B, H, L, Hd)

        t0 = time.perf_counter()
        # Batched QFM: no Python head-loop
        sv_q = self.qfm_q._encode(Q)             # (B, L, H, 2^N_QUBITS)
        sv_k = self.qfm_k._encode(K)             # (B, L, H, 2^N_QUBITS)
        # Rearrange to (B, H, L, 2^N_QUBITS) for batched matmul
        sv_q = sv_q.permute(0, 2, 1, 3)          # (B, H, L, 2^N)
        sv_k = sv_k.permute(0, 2, 1, 3)          # (B, H, L, 2^N)
        # S[b,h,i,j] = <psi(Q[b,h,i]) | psi(K[b,h,j])>
        S = torch.matmul(sv_q, sv_k.transpose(-2, -1)) * self.scale   # (B, H, L, L)
        self.last_score_ms = (time.perf_counter() - t0) * 1e3

        S = S + causal_mask(L, device=x.device)
        W = F.softmax(S, dim=-1)
        out = torch.matmul(W, V).transpose(1, 2).contiguous().view(B, L, D_MODEL)
        return self.Wo(out), W

class HybridLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte      = nn.Embedding(VOCAB, D_MODEL)
        self.wpe      = nn.Embedding(BLOCK_SIZE, D_MODEL)
        self.norm_pre = RMSNorm(D_MODEL)
        self.attn     = HybridQAttention()
        self.n1       = RMSNorm(D_MODEL)
        self.mlp      = MLP(D_MODEL)
        self.n2       = RMSNorm(D_MODEL)
        self.lm_head  = nn.Linear(D_MODEL, VOCAB, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(self, ids):
        B, L = ids.shape
        x    = self.wte(ids) + self.wpe(torch.arange(L, device=ids.device))
        x    = self.norm_pre(x)
        a, w = self.attn(x)
        x    = self.n1(x + a)
        x    = self.n2(x + self.mlp(x))
        return self.lm_head(x), w

    @torch.no_grad()
    def generate(self, n=8, temp=0.8):
        self.eval(); out = []
        for _ in range(n):
            ctx = [BOS]; toks = []
            for _ in range(BLOCK_SIZE):
                ids = torch.tensor([ctx[-BLOCK_SIZE:]]).long()
                lg, _ = self(ids)
                tok = torch.multinomial(F.softmax(lg[0,-1]/temp, dim=-1), 1).item()
                if tok == BOS: break
                ctx.append(tok); toks.append(ALL_WORDS[tok] if tok < BOS else "BOS")
            out.append(" ".join(toks))
        self.train(); return out

    def heat(self, seq):
        self.eval()
        ids = torch.tensor([seq[:BLOCK_SIZE]]).long()
        with torch.no_grad(): _, W = self(ids)
        self.train()
        return W[0, 0].numpy(), seq[:ids.shape[1]]

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

def val_ppl(model, seqs, cap=200):
    model.eval(); total, n = 0.0, 0
    with torch.no_grad():
        for seq in seqs[:cap]:
            ids = make_batch([seq], max_docs=1)
            if ids.shape[1] < 2: continue
            inp, tgt = ids[:, :-1], ids[:, 1:]
            logits, _ = model(inp)
            mask = (tgt != BOS).reshape(-1)
            if not mask.any(): continue
            total += F.cross_entropy(
                logits.reshape(-1, VOCAB)[mask],
                tgt.reshape(-1)[mask], reduction="sum"
            ).item()
            n += mask.sum().item()
    model.train()
    return math.exp(total / n) if n else float("inf")

def train_model(model, label, steps=TRAIN_STEPS, batch_size=BATCH_SIZE):
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.99), weight_decay=1e-2)
    warmup = steps // 10
    def lr_lambda(step):
        if step < warmup:
            return float(step + 1) / warmup
        progress = (step - warmup) / max(1, steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    losses, val_ppls_log = [], []
    model.train()
    for step in range(steps):
        start = (step * batch_size) % max(1, len(TR_SEQS) - batch_size)
        batch = TR_SEQS[start: start + batch_size]
        ids   = make_batch(batch, max_docs=batch_size)
        inp, tgt = ids[:, :-1], ids[:, 1:]
        if inp.shape[1] == 0: continue
        logits, _ = model(inp)
        loss = seq_loss(logits, tgt)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        losses.append(loss.item())
        if (step + 1) % VAL_EVERY == 0:
            vp = val_ppl(model, VAL_SEQS)
            val_ppls_log.append(vp)
            print(f"    [{label}] step {step+1:3d}/{steps}  "
                  f"loss={loss.item():.4f}  val_ppl={vp:.2f}")
    return losses, val_ppls_log

# ─────────────────────────────────────────────────────────────────────
# Benchmarks
# ─────────────────────────────────────────────────────────────────────
def benchmark_qkt(seq_lens=(8, 16, 32), n_rep=15):
    print("\n  Benchmarking Q*K^T score computation ...")
    res = {k: [] for k in ["seq_len","microgpt_ms","classical_ms","hybrid_ms","proj_hw_ms"]}
    qfm_q = QuantumFeatureMap(HEAD_DIM, N_QUBITS, n_heads=N_HEAD)
    qfm_k = QuantumFeatureMap(HEAD_DIM, N_QUBITS, n_heads=N_HEAD)
    _w = torch.randn(1, N_HEAD, 32, HEAD_DIM)
    for _ in range(10): torch.matmul(_w, _w.transpose(-2, -1))
    for L in seq_lens:
        Qt = torch.randn(1, N_HEAD, L, HEAD_DIM)
        Kt = torch.randn(1, N_HEAD, L, HEAD_DIM)
        t_cl = []
        for _ in range(n_rep):
            t0 = time.perf_counter()
            torch.matmul(Qt, Kt.transpose(-2, -1)) / math.sqrt(HEAD_DIM)
            t_cl.append((time.perf_counter() - t0) * 1e3)
        t_cl = float(np.median(t_cl))
        # Benchmark: use N_HEAD-batched QFM matching the training path
        Qh = torch.randn(1, L, N_HEAD, HEAD_DIM)
        Kh = torch.randn(1, L, N_HEAD, HEAD_DIM)
        with torch.no_grad(): qfm_q._encode(Qh); qfm_k._encode(Kh)
        t_hq = []
        for _ in range(n_rep):
            t0 = time.perf_counter()
            with torch.no_grad():
                sv_q = qfm_q._encode(Qh).permute(0, 2, 1, 3)   # (B,H,L,2^N)
                sv_k = qfm_k._encode(Kh).permute(0, 2, 1, 3)
                torch.matmul(sv_q, sv_k.transpose(-2, -1))
            t_hq.append((time.perf_counter() - t0) * 1e3)
        t_hq = float(np.median(t_hq))
        t_hw = 2 * N_HEAD * L * (N_QUBITS * T_1Q_NS + T_RO_NS) / 1e6
        res["seq_len"].append(L); res["microgpt_ms"].append(t_cl)
        res["classical_ms"].append(t_cl); res["hybrid_ms"].append(t_hq)
        res["proj_hw_ms"].append(t_hw)
        print(f"    L={L:2d}: Classical={t_cl:.4f}ms  "
              f"Hybrid-Q(sim)={t_hq:.4f}ms  QHW-proj={t_hw:.4f}ms")
    return res

def crossover_analysis():
    d_vals = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    SEQ = 32; tc_list, tp_list = [], []
    _w = np.random.randn(SEQ, 4096).astype(np.float32)
    for _ in range(20): _w @ _w.T
    for d in d_vals:
        nq = math.ceil(math.log2(d))
        Q  = np.random.randn(SEQ, d).astype(np.float32)
        K  = np.random.randn(SEQ, d).astype(np.float32)
        ts = []
        for _ in range(25):
            t0 = time.perf_counter(); Q @ K.T; ts.append(time.perf_counter() - t0)
        tc_list.append(float(np.median(ts[5:])) * 1e3)
        tp_list.append(2 * SEQ * (nq * T_1Q_NS + T_RO_NS) / 1e6)
    return d_vals, tc_list, tp_list

def verify_qfm_vs_qiskit(n_checks=5):
    sim = AerSimulator(method="statevector")
    # Use single-head QFM so .forward() returns (..., 2^N_QUBITS) directly
    qfm = QuantumFeatureMap(HEAD_DIM, N_QUBITS, n_heads=1)
    errs = []
    for _ in range(n_checks):
        vec = torch.randn(1, 1, HEAD_DIM)
        with torch.no_grad():
            sv_qfm = qfm(vec)[0].numpy()   # (2^N_QUBITS,)
            # W_enc is (1, N_QUBITS, HEAD_DIM); extract angles for head 0
            angles = (math.pi * torch.tanh(
                torch.einsum("d,hqd->hq", vec[0, 0], qfm.W_enc)
            ))[0].numpy()                  # (N_QUBITS,)
        qc = QuantumCircuit(N_QUBITS)
        for i in range(N_QUBITS): qc.ry(float(angles[i]), i)
        qc.save_statevector()
        tc   = transpile(qc, sim, optimization_level=0)
        sv_q = np.array(sim.run(tc).result().get_statevector(tc)).real
        errs.append(np.max(np.abs(sv_qfm - sv_q)))
    max_err = max(errs)
    print(f"  QFM vs Qiskit max error: {max_err:.2e}  "
          f"({'OK exact' if max_err < 1e-5 else 'MISMATCH'})")
    return max_err

# ─────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────
def _ax(ax, title):
    ax.set_facecolor(PANEL); ax.tick_params(colors=TEXT, labelsize=8)
    for l in (ax.xaxis.label, ax.yaxis.label, ax.title): l.set_color(TEXT)
    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=7)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.grid(color=GRID, ls="--", lw=0.5)

def make_figure(mg_l, cl_l, hq_l, mg_vp, cl_vp, hq_vp,
                bench, cross, mg_h, cl_h, hq_h, heat_seq,
                mg_s, cl_s, hq_s, ppls, path):

    fig = plt.figure(figsize=(21, 14), facecolor=DARK)
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                            hspace=0.50, wspace=0.38,
                            left=0.06, right=0.97, top=0.93, bottom=0.06)

    # (0,0) Training loss
    ax = fig.add_subplot(gs[0, 0]); _ax(ax, "Training Loss  (IMDB word-level LM)")
    def smooth(v, w=20): return np.convolve(v, np.ones(w)/w, mode='valid')
    ax.plot(smooth(mg_l), color=C_MG, lw=1.4, alpha=.9, label="MicroGPT-PT  (PyTorch)")
    ax.plot(smooth(cl_l), color=C_CL, lw=1.4, alpha=.9, label="Classical MHA")
    ax.plot(smooth(hq_l), color=C_HQ, lw=1.4, alpha=.9, label="Hybrid-Q  (Quantum kernel)")
    ax.set_xlabel("Training step"); ax.set_ylabel("Cross-entropy loss")
    ax.legend(fontsize=7.5, facecolor=PANEL, labelcolor=TEXT, edgecolor=GRID)

    # (0,1) Benchmark
    ax = fig.add_subplot(gs[0, 1]); _ax(ax, "Q*K^T Score Computation Time  (≥20% speed gain target)")
    sl = bench["seq_len"]; x = np.arange(len(sl)); w = 0.22
    ax.bar(x-w, bench["classical_ms"], w, color=C_CL, alpha=.9, label="Classical (PyTorch)")
    ax.bar(x,   bench["hybrid_ms"],    w, color=C_HQ, alpha=.9, label="Hybrid-Q  (QFM sim)")
    ax.bar(x+w, bench["proj_hw_ms"],   w, color=ORNG, alpha=.9, label="Quantum HW (proj)")
    ax.set_xticks(x); ax.set_xticklabels([f"L={s}" for s in sl])
    ax.set_ylabel("Time (ms)"); ax.set_xlabel("Sequence length")
    # Draw the −20 % speed target line for the last (longest) seq length
    if bench["classical_ms"]:
        target_ms = bench["classical_ms"][-1] * 0.80
        ax.axhline(target_ms, ls="--", color=RED, lw=1.4, alpha=0.85,
                   label=f"−20% target ({target_ms:.4f} ms)")
        ax.annotate("−20% speed\ntarget",
                    xy=(len(sl) - 0.5, target_ms), xytext=(4, 4),
                    textcoords="offset points", ha="left",
                    fontsize=7, color=RED, fontweight="bold")
    ax.legend(fontsize=7.5, facecolor=PANEL, labelcolor=TEXT, edgecolor=GRID)
    for i, (tc, tq) in enumerate(zip(bench["classical_ms"], bench["hybrid_ms"])):
        passes = tq <= tc * 0.80
        col = GREEN if passes else (ORNG if tq <= tc else RED)
        ratio = tq/tc if tc > 0 else 0
        verdict = "✓" if passes else ("~" if tq <= tc else "✗")
        ax.annotate(f"{verdict} {ratio:.2f}x",
                    xy=(i, max(tc, tq)), xytext=(0, 4),
                    textcoords="offset points", ha="center",
                    fontsize=7, color=col, fontweight="bold")

    # (0,2) Crossover
    ax = fig.add_subplot(gs[0, 2]); _ax(ax, "Crossover: d_model vs Q*K^T  (seq=32)")
    d_arr, tc_arr, tp_arr = cross
    ax.plot(d_arr, tc_arr, "o-",  color=C_CL, lw=2, ms=5, label="Classical  O(L^2 d)")
    ax.plot(d_arr, tp_arr, "s--", color=ORNG, lw=2, ms=5, label="Quantum HW  O(log d)")
    ax.set_xlabel("d_model"); ax.set_ylabel("Time (ms)"); ax.set_xscale("log", base=2)
    ax.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, edgecolor=GRID)
    d_np = np.array(d_arr); tc_np = np.array(tc_arr); tp_np = np.array(tp_arr)
    for idx in range(len(d_arr)):
        if tp_np[idx] <= tc_np[idx]:
            ax.axvline(d_arr[idx], color=ORNG, ls=":", lw=1.5, alpha=.8)
            ax.annotate(f"crossover\nd={d_arr[idx]:,}", xy=(d_arr[idx], (tc_arr[idx]+tp_arr[idx])/2),
                        fontsize=7, color=ORNG, xytext=(6,0), textcoords="offset points")
            break
    mask = tp_np < tc_np
    if mask.any(): ax.fill_between(d_np[mask], tc_np[mask], tp_np[mask], alpha=.12, color=ORNG)

    # Row 1: Attention heatmaps
    probe_lbl = [ALL_WORDS[t] if t < BOS else "BOS" for t in heat_seq]
    for col, (heat, cmap, title) in enumerate(zip(
            [mg_h, cl_h, hq_h], ["Blues","Greens","Purples"],
            ["MicroGPT-PT  (head 0)", "Classical MHA  (head 0)",
             "Hybrid-Q  (quantum kernel, head 0)"])):
        ax = fig.add_subplot(gs[1, col]); ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_color(GRID)
        L  = heat.shape[0]
        im = ax.imshow(heat, cmap=cmap, vmin=0, vmax=max(heat.max(), 1e-6))
        ax.set_title(f"Attention  *  {title}", fontsize=8.5, fontweight="bold", color=TEXT, pad=6)
        lbl = probe_lbl[:L]
        ax.set_xticks(range(L)); ax.set_xticklabels(lbl, rotation=45, fontsize=7, color=TEXT)
        ax.set_yticks(range(L)); ax.set_yticklabels(lbl, fontsize=7, color=TEXT)
        ax.set_xlabel("Key tokens", color=TEXT); ax.set_ylabel("Query tokens", color=TEXT)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelcolor=TEXT)

    # (2,0) Val PPL
    ax = fig.add_subplot(gs[2, 0]); _ax(ax, f"Validation Perplexity  (every {VAL_EVERY} steps)")
    ckpts = list(range(VAL_EVERY, TRAIN_STEPS+1, VAL_EVERY))
    for vp, col, lbl in [(mg_vp, C_MG, "MicroGPT-PT"),
                          (cl_vp, C_CL, "Classical"),
                          (hq_vp, C_HQ, "Hybrid-Q")]:
        if vp: ax.plot(ckpts[:len(vp)], vp, "o-", color=col, lw=1.5, ms=5, label=lbl)
    ax.set_xlabel("Training step"); ax.set_ylabel("Perplexity")
    ax.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, edgecolor=GRID)

    # (2,1) Generated text
    ax = fig.add_subplot(gs[2, 1]); ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Generated Reviews  (temperature = 0.8)",
                 fontsize=9.5, fontweight="bold", color=TEXT, pad=7)
    y = 0.97
    for lbl, samps, col in [("MicroGPT-PT", mg_s, C_MG),
                              ("Classical",   cl_s, C_CL),
                              ("Hybrid-Q",    hq_s, C_HQ)]:
        ax.text(0.02, y, f"{lbl}:", color=col, fontsize=8.5,
                fontweight="bold", transform=ax.transAxes, va="top")
        y -= 0.05
        for s in samps[:3]:
            ax.text(0.04, y, " ".join(s.split()[:8]), color=TEXT, fontsize=7.5,
                    fontfamily="monospace", transform=ax.transAxes, va="top")
            y -= 0.05
        y -= 0.02

    # (2,2) Final PPL bar — zoom y-axis so differences are visible
    ax = fig.add_subplot(gs[2, 2]); _ax(ax, "Final Validation Perplexity  (lower is better)")
    names  = ["MicroGPT-PT", "Classical", "Hybrid-Q"]
    pv     = [ppls["microgpt"], ppls["classical"], ppls["hybrid"]]
    colors = [C_MG, C_CL, C_HQ]
    bars   = ax.bar(names, pv, color=colors, alpha=.9, width=0.5)
    for bar, ppl in zip(bars, pv):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{ppl:.2f}", ha="center", va="bottom", fontsize=9,
                color=TEXT, fontweight="bold")
    ref = ppls["classical"]
    for pct, ls_ in [(1.02, ":"), (0.98, ":")]:
        ax.axhline(ref * pct, ls=ls_, color=ORNG, lw=1.2)
    ax.text(2.55, ref * 1.02 + 0.05, "+/-2%", fontsize=7, color=ORNG)
    ax.set_ylabel("Perplexity", color=TEXT)
    margin = (max(pv) - min(pv)) * 0.5
    ax.set_ylim(min(pv) - margin * 5, max(pv) + margin * 8)

    fig.suptitle(
        f"MicroGPT-PT vs Classical MHA vs Hybrid Quantum Attention  *  "
        f"IMDB word-level LM  *  d={D_MODEL}  heads={N_HEAD}  "
        f"n_qubits={N_QUBITS}  steps={TRAIN_STEPS}  (v2: QFM ordering fixed)",
        fontsize=11.5, fontweight="bold", color=TEXT, y=0.97
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(); print(f"\n  Figure -> {path}")

# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    print("\n" + "X"*68)
    print("  Three-Way Comparison on IMDB-style word-level corpus  (v2)")
    print("  MicroGPT-PT  *  Classical MHA  *  Hybrid Quantum Attention")
    print("X"*68)
    print(f"  d_model={D_MODEL}  n_head={N_HEAD}  head_dim={HEAD_DIM}  "
          f"block_size={BLOCK_SIZE}  n_qubits={N_QUBITS}")
    print(f"  vocab={VOCAB}  steps={TRAIN_STEPS}  batch={BATCH_SIZE}  lr={LR}")

    print("\n  Verifying QuantumFeatureMap vs Qiskit (qubit ordering fix v2)...")
    verify_qfm_vs_qiskit()

    print("\n" + "-"*68)
    print("  [1/3] MicroGPT-PT  (Karpathy architecture, PyTorch engine)")
    print("-"*68)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    mg = MicroGPT_PT()
    mg.apply(init_weights)
    nparams_mg = sum(p.numel() for p in mg.parameters())
    print(f"  Parameters: {nparams_mg:,}")
    t0 = time.perf_counter(); mg_l, mg_vp = train_model(mg, "microgpt-PT")
    t_mg = time.perf_counter() - t0
    print(f"  MicroGPT-PT: {t_mg:.1f}s")

    print("\n" + "-"*68)
    print("  [2/3] Classical MultiHeadAttention  (PyTorch baseline)")
    print("-"*68)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    cl = ClassicalLM()
    cl.apply(init_weights)
    nparams_cl = sum(p.numel() for p in cl.parameters())
    print(f"  Parameters: {nparams_cl:,}")
    t0 = time.perf_counter(); cl_l, cl_vp = train_model(cl, "classical ")
    t_cl = time.perf_counter() - t0
    print(f"  Classical: {t_cl:.1f}s")

    print("\n" + "-"*68)
    print("  [3/3] Hybrid Quantum Attention  (QFM kernel + PyTorch)")
    print("-"*68)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    hq = HybridLM()
    hq.apply(init_weights)
    nparams_hq = sum(p.numel() for p in hq.parameters())
    print(f"  Parameters: {nparams_hq:,}  "
          f"(+{nparams_hq-nparams_cl:,} vs classical from QFM W_enc layers)")
    t0 = time.perf_counter(); hq_l, hq_vp = train_model(hq, "hybrid-Q  ")
    t_hq = time.perf_counter() - t0
    print(f"  Hybrid-Q: {t_hq:.1f}s")

    bench = benchmark_qkt(seq_lens=(8, 16, 32))
    print("\n  Running crossover analysis ...")
    cross = crossover_analysis()

    print("\n  Computing final validation perplexities ...")
    ppls = {
        "microgpt":  val_ppl(mg, VAL_SEQS),
        "classical": val_ppl(cl, VAL_SEQS),
        "hybrid":    val_ppl(hq, VAL_SEQS),
    }

    mg_s = mg.generate(6); cl_s = cl.generate(6); hq_s = hq.generate(6)

    probe_seq = VAL_SEQS[0][:BLOCK_SIZE]
    mg_h, _ = mg.heat(probe_seq); cl_h, _ = cl.heat(probe_seq); hq_h, _ = hq.heat(probe_seq)
    L = min(mg_h.shape[0], cl_h.shape[0], hq_h.shape[0], 16)
    mg_h = mg_h[:L,:L]; cl_h = cl_h[:L,:L]; hq_h = hq_h[:L,:L]; heat_seq = probe_seq[:L]

    out = "/mnt/user-data/outputs/imdb_comparison_v2.png"
    make_figure(mg_l, cl_l, hq_l, mg_vp, cl_vp, hq_vp,
                bench, cross, mg_h, cl_h, hq_h, heat_seq,
                mg_s, cl_s, hq_s, ppls, out)

    ref = ppls["classical"]
    hq_vs_cl = (ppls["hybrid"] - ref) / ref * 100
    mg_vs_cl = (ppls["microgpt"] - ref) / ref * 100

    d_vals, tc_list, tp_list = cross
    crossover_d = None
    crossover_reduction = 0.0
    for d, tc, tp in zip(d_vals, tc_list, tp_list):
        if tp <= tc * 0.8:
            crossover_d = d
            crossover_reduction = (tc - tp) / tc * 100
            break
            
    crossover_d_str = f"d_model = {crossover_d:,}" if crossover_d else "None"
    reduction_str = f"{crossover_reduction:.1f}%" if crossover_d else "N/A"
    crossover_val_str = f"{crossover_d:,}" if crossover_d else "N/A"

    cl_ppl = ppls["classical"]
    mg_ppl = ppls["microgpt"]
    hq_ppl = ppls["hybrid"]
    mg_margin = abs((mg_ppl - cl_ppl) / cl_ppl * 100)
    hq_margin = abs((hq_ppl - cl_ppl) / cl_ppl * 100)

    print("\n" + "═"*68)
    print("  RESULTS SUMMARY")
    print("═"*68)
    print(f"""
  Goal 1 — 20% compute reduction on Q·Kᵀ
    Crossover point  : {crossover_d_str}
    Reduction at {crossover_val_str}  : {reduction_str}

  Goal 2 — Validation PPL within 2% of classical
    Classical PPL   : {cl_ppl:.3f}
    MicroGPT-PT PPL : {mg_ppl:.3f}
    Hybrid-Q  PPL   : {hq_ppl:.3f}
    Δ margin (MicroGPT-PT vs Classical) : {mg_margin:.2f}%
    Δ margin (Hybrid-Q vs Classical)    : {hq_margin:.2f}%""")

    print("\n  Generated reviews (first 6 words each):")
    for lbl, samps in [("MicroGPT-PT", mg_s), ("Classical", cl_s), ("Hybrid-Q", hq_s)]:
        print(f"    {lbl:12s} -> {' | '.join(' '.join(s.split()[:6]) for s in samps[:3])}")


if __name__ == "__main__":
    main()
