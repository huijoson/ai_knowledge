# 規格說明：Transformer 與現代 LLM 架構

本規格書涵蓋了經典 Transformer (Vaswani et al., 2017) 及其在現代大語言模型（LLM，如 LLaMA、Mistral、GPT-4）中的現代化演進技術。

---

## 1. 模組概述
Transformer 架構已從最初的序列到序列（Seq2Seq）機器翻譯模型，演化為當今主流的僅解碼器（Decoder-Only）架構。現代 LLM 進行了多項優化，以改善訓練穩定性（RMSNorm、SwiGLU、RoPE）並提升推理速度（GQA、KV Cache）。

---

## 2. 數學定義與現代升級

### 2.1 注意力機制變體

#### 1. 多頭注意力 (Multi-Head Attention, MHA)
每個 Query 頭都擁有自己獨立的 Key 和 Value 頭：
$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

#### 2. 多查詢注意力 (Multi-Query Attention, MQA)
所有 Query 頭共享同一個 Key 和 Value 頭，以在解碼生成（Inference）時節省記憶體空間：
$$\text{head}_i = \text{Attention}(Q W_i^Q, K W^K, V W^V)$$

#### 3. 分組查詢注意力 (Grouped-Query Attention, GQA)
將 Query 頭分組，每一組共享同一個 Key 和 Value 頭。這是 MHA 與 MQA 之間的折衷方案（LLaMA 3 的標準配置）：
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

### 2.2 歸一化層變體 (Normalization)

#### 1. 層歸一化 (Layer Normalization, LayerNorm)
$$\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$
其中 $\mu$ 與 $\sigma^2$ 是沿著特徵維度（最後一維）計算的平均值與變異數。

#### 2. 均方根歸一化 (RMSNorm)
藉由去除平均值計算來簡化 LayerNorm，在保持相似性能的同時節省大量計算開銷（LLaMA 的標準配置）：
$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \odot \gamma$$
$$\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}$$

---

### 2.3 位置編碼 (Positional Embeddings)

#### 1. 正弦位置編碼 (Sinusoidal Positional Encoding)
將靜態的正弦和餘弦波坐標直接加算到輸入嵌入（Embeddings）上。

#### 2. 旋轉位置編碼 (Rotary Position Embedding, RoPE)
在複數平面上對 Query 和 Key 進行旋轉，使注意力機制中的點積自然地捕捉相對距離（LLaMA 的標準配置）：
$$\text{RoPE}(x_m) = R_{\Theta, m}^d x_m$$
其中 $R_{\Theta, m}^d$ 是位置 $m$ 的分塊對角旋轉矩陣。

---

### 2.4 前饋網路變體 (FFN)

#### 1. 經典 FFN
$$\text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2$$

#### 2. SwiGLU (Swish 門控線性單元)
使用 Swish 激活函數（$\text{Swish}(x) = x \cdot \sigma(\beta x)$）建構門控結構，展現出更優越的經驗成效（LLaMA 的標準配置）：
$$\text{SwiGLU}(x) = (\text{Swish}(x W_{gate}) \odot x W_{up}) W_{down}$$

---

## 3. 張量維度 (Tensor Shapes)

| 變數 / 參數 | 維度 (Shape) | 描述 |
| :--- | :--- | :--- |
| `x` (輸入序列) | `(B, L, d_model)` | 批次大小 $B$，序列長度 $L$，嵌入維度 $d_{model}$ |
| `q` (GQA 中的 Queries) | `(B, H_q, L, d_k)` | $H_q$ 為 Query 頭的個數 |
| `k`, `v` (Keys, Values) | `(B, H_kv, L, d_k)`| $H_{kv}$ 為 KV 頭的個數 (GQA 中: $H_{kv} < H_q$) |
| `RoPE coefficients` | `(L, d_k)` | 每個位置對應的旋轉角度參數 |

---

## 4. 驗證要求 (單元測試)

為確保各組件實作正確：
1.  **GQA 廣播對齊**：驗證 Query 頭是否能正確與分組共享的 Key/Value 頭進行點積對齊。
2.  **RMSNorm 縮放特性**：驗證 RMSNorm 輸出在應用學得的 $\gamma$ 之前，其均方根值是否大約為 1。
3.  **RoPE 相對距離特性**：驗證兩個旋轉後向量的點積 $q_m^T k_n$ 是否僅取決於它們的相對距離 $m - n$。
4.  **SwiGLU 門控運作**：驗證 SwiGLU 的輸出維度正確，且門控分支能正常調節資訊流。
