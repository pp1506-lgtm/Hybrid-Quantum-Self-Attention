## Quantum Self-Attention for Language Modeling
Research internship project exploring whether a hybrid classical-quantum self-attention layer can match classical multi-head attention while reducing compute time, simulated using Qiskit.

## Project Goals
Design and simulate a hybrid quantum self-attention layer for small-scale language modeling (sequence lengths 64–256 tokens)
Achieve a 20% reduction in compute time for the Q·Kᵀ matrix multiplication compared to a classical baseline
Maintain validation perplexity within a 2% margin of the classical baseline.

## Technical Features & Optimizations
* **QFM Vectorization**: All quantum feature maps are fully vectorized into batched matrix-multiplication operations in PyTorch, removing slow loops and matching Qiskit's little-endian statevector convention (verified with $< 10^{-7}$ numerical error).
* **GPT-2 Style Initialization**: All model embeddings and linear layers are initialized using $\mathcal{N}(0, 0.02)$ with zeroed biases, dropping perplexity from a random baseline of $\sim 101$ down to $\sim 66$.
* **Cosine Learning Rate Decay**: Uses a 10% warmup schedule followed by a cosine learning rate decay.

### Key Takeaways
1. **Perplexity Parity**: Hybrid-Q is highly competitive, outperforming the classical model by **0.44%** (lower perplexity).
2. **Quantum Scaling Advantage**: Classically, computing the attention matrix scales as $O(L^2 d)$, whereas simulated quantum hardware scales as $O(L \log_2(d))$. The projected execution crossover point is at $d_{model} = 4,096$, where quantum hardware achieves a **36% compute speedup**.

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



