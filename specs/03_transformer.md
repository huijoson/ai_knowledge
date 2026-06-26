# Specification: The Transformer & Modern LLM Architecture

This specification covers the original Transformer (Vaswani et al., 2017) and its modern evolutions used in LLMs like LLaMA, Mistral, and GPT-4.

---

## 1. Module Overview
The Transformer architecture has evolved from a sequence-to-sequence machine translation model into decoder-only models (like GPT) and encoder-decoder models. Modern LLMs have optimization tweaks to improve training stability (RMSNorm, SwiGLU, RoPE) and inference speed (GQA, KV Cache).

---

## 2. Mathematical Definition & Modern Upgrades

### 2.1 Attention Variants

#### 1. Multi-Head Attention (MHA)
Every Query head has its own Key and Value head:
$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

#### 2. Multi-Query Attention (MQA)
All Query heads share a single Key and Value head to save memory during decoding:
$$\text{head}_i = \text{Attention}(Q W_i^Q, K W^K, V W^V)$$

#### 3. Grouped-Query Attention (GQA)
Query heads are grouped, and each group shares a single Key and Value head. This is a middle ground between MHA and MQA (used in LLaMA 3):
$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_{group(i)}^K, V W_{group(i)}^V)$$

```
Multi-Head Attention (MHA)    Grouped-Query (GQA)         Multi-Query (MQA)
    Q1 Q2 Q3 Q4                   Q1 Q2 Q3 Q4                 Q1 Q2 Q3 Q4
    │  │  │  │                    └─┬─┘  └─┬─┘                └───┬───┘
    ▼  ▼  ▼  ▼                      ▼      ▼                      ▼
    K1 K2 K3 K4                     K1     K2                     K1
    V1 V2 V3 V4                     V1     V2                     V1
```

---

### 2.2 Normalization Variants

#### 1. Layer Normalization (LayerNorm)
$$\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$
Where $\mu$ is the mean and $\sigma^2$ is the variance along the feature dimension.

#### 2. RMSNorm (Root Mean Square Normalization)
Simplifies LayerNorm by removing the mean calculation, saving computation while maintaining similar performance (used in LLaMA):
$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \odot \gamma$$
$$\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}$$

---

### 2.3 Positional Embeddings

#### 1. Sinusoidal Positional Encoding
Adds static sine and cosine wave coordinates to input embeddings.

#### 2. Rotary Position Embedding (RoPE)
Applies a rotation to the Queries and Keys in the complex plane, allowing the dot-product attention to naturally capture relative distance (used in LLaMA):
$$\text{RoPE}(x_m) = R_{\Theta, m}^d x_m$$
Where $R_{\Theta, m}^d$ is a block-diagonal rotation matrix for position $m$.

---

### 2.4 Feed-Forward Network (FFN) Variants

#### 1. Vanilla FFN
$$\text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2$$

#### 2. SwiGLU (Swish Gated Linear Unit)
Uses the Swish activation ($\text{Swish}(x) = x \cdot \sigma(\beta x)$) in a gated structure, showing superior empirical results (used in LLaMA):
$$\text{SwiGLU}(x) = (\text{Swish}(x W_{gate}) \odot x W_{up}) W_{down}$$

---

## 3. Tensor Shapes

| Parameter / Variable | Shape | Description |
| :--- | :--- | :--- |
| `x` (Input sequence) | `(B, L, d_model)` | Batch size $B$, Sequence length $L$, Embedding dim $d_{model}$ |
| `q` (Queries in GQA) | `(B, H_q, L, d_k)` | $H_q$ is number of query heads |
| `k`, `v` (Keys, Values) | `(B, H_kv, L, d_k)`| $H_{kv}$ is number of KV heads (GQA: $H_{kv} < H_q$) |
| `RoPE coefficients` | `(L, d_k)` | Rotation angles per position |

---

## 4. Verification Requirements (Tests)

To verify the components:
1.  **GQA Broadcasting**: Verify that query heads correctly align with group-shared key/value heads.
2.  **RMSNorm Scaling**: Verify that RMSNorm output has a root-mean-square value of approximately 1 (before scale $\gamma$).
3.  **RoPE Relative Properties**: Verify that the dot product of two rotated vectors $q_m^T k_n$ depends only on their relative distance $m - n$.
4.  **SwiGLU Gates**: Verify that SwiGLU output dimensions are correct and activation gating operates properly.
