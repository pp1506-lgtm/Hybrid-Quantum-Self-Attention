# Hybrid Quantum Attention Language Model (Comparison Benchmark)

This repository benchmarks and compares a PyTorch causal language model across three attention architectures on an IMDB-style word-level sentiment movie review corpus (3,000 reviews, structured vocabulary).

## Architecture Comparison

1. **MicroGPT-PT**: A native PyTorch module implementation of Andrej Karpathy's `microgpt.py` architecture (1 layer, Multi-Head Attention, MLP, RMSNorm, residuals, weight-tied lm_head).
2. **Classical MHA**: Standard PyTorch multi-head scaled dot-product attention serving as the baseline.
3. **Hybrid-Q Attention**: A hybrid quantum-classical multi-head attention network where Query/Key/Value projections are classical, but query-key similarity scores are computed as quantum state overlaps $\langle \psi(Q) | \psi(K) \rangle$ using an exact, fully vectorized $RY$-circuit angle encoding simulation.

---

## Technical Features & Optimizations

* **QFM Vectorization**: All quantum feature maps are fully vectorized into batched matrix-multiplication operations in PyTorch, removing slow loops and matching Qiskit's little-endian statevector convention (verified with $< 10^{-7}$ numerical error).
* **GPT-2 Style Initialization**: All model embeddings and linear layers are initialized using $\mathcal{N}(0, 0.02)$ with zeroed biases, dropping perplexity from a random baseline of $\sim 101$ down to $\sim 66$.
* **Cosine Learning Rate Decay**: Uses a 10% warmup schedule followed by a cosine learning rate decay.

---

## Results Summary

```text
════════════════════════════════════════════════════════════════════
  RESULTS SUMMARY
════════════════════════════════════════════════════════════════════

  Goal 1 — 20% compute reduction on Q·Kᵀ
    Crossover point  : d_model = 4,096
    Reduction at 4,096  : 36.0%

  Goal 2 — Validation PPL within 2% of classical
    Classical PPL   : 66.849
    MicroGPT-PT PPL : 66.849
    Hybrid-Q  PPL   : 66.557
    Δ margin (MicroGPT-PT vs Classical) : 0.00%
    Δ margin (Hybrid-Q vs Classical)    : 0.44%
```

### Key Takeaways
1. **Perplexity Parity**: Hybrid-Q is highly competitive, outperforming the classical model by **0.44%** (lower perplexity).
2. **Quantum Scaling Advantage**: Classically, computing the attention matrix scales as $O(L^2 d)$, whereas simulated quantum hardware scales as $O(L \log_2(d))$. The projected execution crossover point is at $d_{model} = 4,096$, where quantum hardware achieves a **36% compute speedup**.

---

## Getting Started

### Prerequisites
Install dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Run the Benchmark
Execute the comparison script to train the models, benchmark execution speed, verify correctness against Qiskit, and generate comparison plots:
```bash
python comparisionafter.py
```
This produces an output comparison chart at `outputs/imdb_comparison_v2.png`.
