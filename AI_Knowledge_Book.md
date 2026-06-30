# 從零手刻 LLM：Transformer 核心原理互動電子書

歡迎閱讀這本互動電子書！本書專為希望藉由「從零手寫實作（From Scratch）」真正吃透現代大語言模型（LLM）底層機制的自學者所寫。

我們不只停留在 API 使用，也不只背公式。本書會用 **白話直覺 → 數學公式 → 手算範例 → Tensor Shape → 互動視覺化 → PyTorch 對照 → 從零實作 → 單元測試驗證** 的方式，把 LLM 的核心架構一步步拆開。

---

## 📖 本書大綱

> 本書採用 **LLM-first** 順序：先建立 Transformer / LLM 主線，再把 RNN、CNN 放到補充篇作為歷史脈絡。

*   [導讀：零數學直覺導引 — 用生活白話解碼硬核數學](#導讀零數學直覺導引--用生活白話解碼硬核數學)
*   [第 0 章：LLM 底層架構學習地圖 — 從文字到下一個 Token](#第-0-章llm-底層架構學習地圖--從文字到下一個-token)
*   [第 1 章：Tensor、矩陣與 Shape 直覺 — 看懂 LLM 的資料形狀](#第-1-章tensor矩陣與-shape-直覺--看懂-llm-的資料形狀)
*   [第 2 章：Linear Layer 與基本神經網路運算 — 投影、升維與降維](#第-2-章linear-layer-與基本神經網路運算--投影升維與降維)
*   [第 3 章：Autograd 與反向傳播 — 模型如何學習](#第-3-章autograd-與反向傳播--模型如何學習)
*   [第 4 章：Optimizer 與 AdamW — 參數如何被更新](#第-4-章optimizer-與-adamw--參數如何被更新)
*   [第 5 章：Tokenization 與 Embedding — 文字如何變成向量](#第-5-章tokenization-與-embedding--文字如何變成向量)
*   [第 6 章：Attention 直覺與 Scaled Dot-Product Attention — 模型如何決定要看哪裡](#第-6-章attention-直覺與-scaled-dot-product-attention--模型如何決定要看哪裡)
*   [第 7 章：Multi-Head Attention — 用多個視角理解上下文](#第-7-章multi-head-attention--用多個視角理解上下文)
*   [第 8 章：Causal Mask 與 Decoder-only LLM — 為什麼不能偷看未來](#第-8-章causal-mask-與-decoder-only-llm--為什麼不能偷看未來)
*   [第 9 章：RMSNorm、Residual Connection 與 Pre-Norm — 讓深層模型穩定訓練](#第-9-章rmsnormresidual-connection-與-pre-norm--讓深層模型穩定訓練)
*   [第 10 章：Feed Forward Network 與 SwiGLU — Transformer 的非線性表達能力](#第-10-章feed-forward-network-與-swiglu--transformer-的非線性表達能力)
*   [第 11 章：RoPE 旋轉位置編碼 — 讓模型知道相對位置](#第-11-章rope-旋轉位置編碼--讓模型知道相對位置)
*   [第 12 章：KV Cache 與 LLM 推理加速 — 不重算過去的祕密](#第-12-章kv-cache-與-llm-推理加速--不重算過去的祕密)
*   [第 13 章：Mini Decoder-only LLM 組裝 — 從元件到完整模型](#第-13-章mini-decoder-only-llm-組裝--從元件到完整模型)
*   [第 14 章：測試驅動理解 LLM — 用 pytest 驗證你的理解](#第-14-章測試驅動理解-llm--用-pytest-驗證你的理解)
*   [補充 A：RNN 與 LSTM — 序列模型的歷史脈絡](#補充-arnn-與-lstm--序列模型的歷史脈絡)
*   [補充 B：CNN 與局部特徵 — 卷積如何看見空間結構](#補充-bcnn-與局部特徵--卷積如何看見空間結構)

---

## 導讀：零數學直覺導引 — 用生活白話解碼硬核數學

如果你看到數學公式就頭痛，請別擔心！深度學習與 LLM 的數學符號本質上只是**用來精確描述資料如何流動與變形的簡寫**。

在進入硬核公式前，我們先用「白話翻譯官」把這些數學符號與概念翻譯成生活常識。

### 0.1 數學符號白話翻譯對照表

| 數學符號 | 英文名稱 | 生活白話解釋 | Python 程式碼對照 |
| :--- | :--- | :--- | :--- |
| $\sum_{i=1}^N x_i$ | Summation | **全部加起來**。把一堆數字加總。 | `sum(x)` |
| $\odot$ | Hadamard Product | **對應位置相乘**。兩個一樣形狀的 tensor 逐元素相乘。 | `a * b` |
| $\sigma(x)$ | Sigmoid | **開關閥門**。把數字壓到 0 到 1 之間。 | `torch.sigmoid(x)` |
| $\nabla_\theta L$ | Gradient | **下坡最陡方向**。指引參數往哪裡改，loss 會下降最快。 | `theta.grad` |
| $W^T$ | Transpose | **橫豎對調**。把矩陣列與行交換。 | `W.T` 或 `W.t()` |
| $A \cdot B$ / $AB$ | Matrix Multiply | **多重條件配方計算**。輸入與權重配對加總。 | `A @ B` 或 `torch.matmul(A, B)` |
| $\text{softmax}(x)$ | Softmax | **把分數變成比例**。讓每個候選項的權重加總為 1。 | `torch.softmax(x, dim=-1)` |

---

### 0.2 LLM 核心概念生活比喻

#### 1. Tensor Shape $\rightarrow$ 「貨物箱上的尺寸標籤」
LLM 裡的每份資料都有形狀，例如 `(B, T, C)`。你可以把它想成貨物箱標籤：這批資料有幾箱（batch）、每箱幾個 token、每個 token 有幾個特徵。

#### 2. Linear Layer $\rightarrow$ 「配方調配機」
線性層就像調酒配方：不同輸入特徵乘上不同權重，再加上一點偏置，調出新的表示。

#### 3. Autograd $\rightarrow$ 「拼圖的組裝與倒帶推演」
前向傳播像組裝樂高，反向傳播像倒帶追查每個零件對最後錯誤的責任。

#### 4. Optimizer $\rightarrow$ 「在濃霧中滾下山的小球」
Optimizer 會根據梯度調整參數。AdamW 加入動量、自適應步伐與 decoupled weight decay，讓訓練更穩定。

#### 5. Tokenization / Embedding $\rightarrow$ 「文字轉換成座標」
文字不能直接被神經網路計算，所以要先切成 token，再查表變成向量座標。

#### 6. Attention $\rightarrow$ 「圖書館的條碼檢索系統」
Query 是你要找什麼，Key 是每本書的標籤，Value 是書的內容。Attention 根據相似度決定該讀哪些內容。

#### 7. Causal Mask $\rightarrow$ 「考試時不能偷看後面的答案」
Decoder-only LLM 在預測下一個 token 時，只能看已經出現的 token，不能看未來。

#### 8. RoPE $\rightarrow$ 「旋轉指北針」
RoPE 讓向量隨位置旋轉，不同 token 之間的相對距離會反映在旋轉角度差中。

#### 9. KV Cache $\rightarrow$ 「開卷考試的草稿紙」
生成新 token 時，已經算過的 Key / Value 不必重算，可以快取起來反覆使用。

---

## 第 0 章：LLM 底層架構學習地圖 — 從文字到下一個 Token

### 0.1 這章要解決什麼問題？

在開始手刻前，我們先建立全局地圖：LLM 並不是魔法，而是一個把「前文 token」轉成「下一個 token 機率分布」的巨大函數。

### 0.2 LLM 的核心資料流

```text
文字
 → Tokenizer
 → Token IDs
 → Embedding Vectors
 → Transformer Block × N
 → Final Norm
 → LM Head
 → Logits
 → Softmax
 → Next Token
```

### 0.3 Decoder-only LLM 的最小組件

一個 GPT-like Decoder-only LLM 至少包含：

1. **Tokenizer**：把文字切成 token 並轉成 id。
2. **Embedding Table**：把 token id 查表成向量。
3. **Transformer Blocks**：重複堆疊的核心計算單元。
4. **Causal Mask**：限制模型只能看過去。
5. **Final RMSNorm**：輸出前的穩定化。
6. **LM Head**：把 hidden state 投影回 vocabulary 分數。
7. **Softmax / Sampling**：把 logits 轉成機率並選出下一個 token。

### 0.4 互動視覺化

<div id="llm-flow-visualizer" class="interactive-visualizer"></div>

### 0.5 本章自我檢查

- LLM 的輸入與輸出分別是什麼？
- 為什麼 LLM 可以被看成 next-token prediction machine？
- Transformer block 在整個資料流中的位置在哪裡？

---

## 第 1 章：Tensor、矩陣與 Shape 直覺 — 看懂 LLM 的資料形狀

### 1.1 這章要解決什麼問題？

LLM 裡幾乎所有錯誤都可以先從 shape 檢查開始。只要看懂資料形狀，就能更容易理解 Linear、Attention、KV Cache 與整個模型 forward pass。

### 1.2 常見維度符號

```text
B = Batch Size，一次處理幾筆資料
T = Sequence Length，每筆資料有幾個 token
C = Hidden Size / Embedding Dimension，每個 token 的向量維度
H = Number of Attention Heads，注意力頭數
D = Head Dimension，每個 head 的維度，通常 D = C / H
V = Vocabulary Size，詞彙表大小
```

### 1.3 LLM 常見 Shape

| 名稱 | Shape | 說明 |
|---|---|---|
| `input_ids` | `(B, T)` | 每個 token 的整數 id |
| `embedding` | `(B, T, C)` | 每個 token 查表後的向量 |
| `q/k/v` | `(B, T, C)` | 線性投影後的 Q/K/V |
| `q_heads` | `(B, H, T, D)` | 拆成多個 attention head |
| `attention_scores` | `(B, H, T, T)` | 每個 token 對每個 token 的注意力分數 |
| `logits` | `(B, T, V)` | 每個位置對整個 vocabulary 的分數 |

### 1.4 Shape 例子

假設：

```text
B = 2
T = 4
C = 8
H = 2
D = 4
V = 100
```

則：

```text
input_ids:        (2, 4)
embeddings:       (2, 4, 8)
q_heads:          (2, 2, 4, 4)
attention_scores: (2, 2, 4, 4)
logits:           (2, 4, 100)
```

### 1.5 互動視覺化

<div id="tensor-shape-visualizer" class="interactive-visualizer"></div>

### 1.6 本章自我檢查

- `(B, T, C)` 分別代表什麼？
- 為什麼拆 head 後會變成 `(B, H, T, D)`？
- `logits` 的最後一維為什麼是 vocabulary size？

---

## 第 2 章：Linear Layer 與基本神經網路運算 — 投影、升維與降維

### 2.1 這章要解決什麼問題？

Transformer 裡大量使用 Linear Layer。Q projection、K projection、V projection、output projection、FFN up/down/gate projection、LM Head，本質上都離不開線性投影。

### 2.2 線性投影層的矩陣計算

線性層的作用是將輸入特徵 $X$ 乘以權重矩陣 $W$ 並加上偏置 $b$：

$$Y = X W^T + b$$

#### 📐 維度拆解說明

假設輸入有 2 個 token，每個 token 是 3 維向量，要投影到 4 維空間：

*   **輸入 $X$**：維度為 `(2, 3)`
*   **權重 $W$**：維度為 `(4, 3)`
*   **偏置 $b$**：維度為 `(4)`
*   **計算 $X W^T$**：`(2, 3) @ (3, 4) → (2, 4)`

在 LLM 中通常會多一個 batch 維度：

```text
X: (B, T, in_features)
W: (out_features, in_features)
Y: (B, T, out_features)
```

---

### 2.3 🧐 手動算一遍：Linear Layer

設：

$$X = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}, \quad W = \begin{pmatrix} 0.5 & 1.5 \\ -1.0 & 2.0 \end{pmatrix}, \quad b = \begin{pmatrix} 0.1 \\ -0.2 \end{pmatrix}$$

計算 $Y = X W^T + b$：

1.  **先算 $W^T$**：

$$W^T = \begin{pmatrix} 0.5 & -1.0 \\ 1.5 & 2.0 \end{pmatrix}$$

2.  **進行矩陣相乘 $X W^T$**：

*   第一列第一行：$1 \times 0.5 + 2 \times 1.5 = 3.5$
*   第一列第二行：$1 \times (-1.0) + 2 \times 2.0 = 3.0$
*   第二列第一行：$3 \times 0.5 + 4 \times 1.5 = 7.5$
*   第二列第二行：$3 \times (-1.0) + 4 \times 2.0 = 5.0$

$$X W^T = \begin{pmatrix} 3.5 & 3.0 \\ 7.5 & 5.0 \end{pmatrix}$$

3.  **加上偏置 $b$**：

$$Y = \begin{pmatrix} 3.6 & 2.8 \\ 7.6 & 4.8 \end{pmatrix}$$

### 2.4 💻 PyTorch 與從零實作對照

你可以在 `src/nn/linear.py` 中實作：

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.matmul(x, self.weight.t()) + self.bias
```

### 2.5 GELU 激活函數

GELU 將輸入乘以高斯累積分布機率：

$$\text{GELU}(x) \approx 0.5x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \left(x + 0.044715 x^3\right)\right)\right)$$

```python
class CustomGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        const = math.sqrt(2.0 / math.pi)
        inner = const * (x + 0.044715 * torch.pow(x, 3))
        return 0.5 * x * (1.0 + torch.tanh(inner))
```

### 2.6 互動視覺化

<div id="matrix-multiplication-visualizer" class="interactive-visualizer"></div>

### 2.7 本章自我檢查

- 為什麼 Linear Layer 可以改變 hidden size？
- Transformer 裡哪些地方會用到 Linear Layer？
- `W` 與 `W^T` 的 shape 為什麼要特別注意？

---

## 第 3 章：Autograd 與反向傳播 — 模型如何學習

### 3.1 這章要解決什麼問題？

前面章節處理 forward 計算；這一章要理解模型如何透過 loss 與 gradient 知道參數該怎麼改。

### 3.2 計算圖的數學原理

任何複雜函數都可以分解為一系列基礎運算。以函數 $z = (x + y) \times w$ 為例：

1.  **定義輸入與中間節點**：
    *   輸入：$x = 2$, $y = 3$, $w = 4$
    *   加法節點：$u = x + y = 5$
    *   乘法節點：$z = u \times w = 20$
2.  **鏈式法則的反向流動**：

$$\frac{\partial z}{\partial x} = \frac{\partial z}{\partial u} \cdot \frac{\partial u}{\partial x} = 4 \times 1 = 4$$

---

### 3.3 🧐 手動算一遍：Autograd

設：

*   $x = 3.0$
*   $y = 5.0$
*   $w = 2.0$

**前向傳播**：

1.  $u = x + y = 8.0$
2.  $z = u \times w = 16.0$

**反向傳播**：

1.  起點梯度 $\frac{\partial z}{\partial z} = 1.0$
2.  $\frac{\partial z}{\partial w} = u = 8.0$
3.  $\frac{\partial z}{\partial u} = w = 2.0$
4.  $\frac{\partial z}{\partial x} = 2.0$
5.  $\frac{\partial z}{\partial y} = 2.0$

### 3.4 💻 PyTorch 程式碼驗證

```python
import torch

x = torch.tensor(3.0, requires_grad=True)
y = torch.tensor(5.0, requires_grad=True)
w = torch.tensor(2.0, requires_grad=True)

u = x + y
z = u * w
z.backward()

print(x.grad) # tensor(2.0)
print(y.grad) # tensor(2.0)
print(w.grad) # tensor(8.0)
```

### 3.5 互動視覺化

<div id="autograd-visualizer" class="interactive-visualizer"></div>

### 3.6 本章自我檢查

- forward pass 與 backward pass 差在哪？
- chain rule 在反向傳播中扮演什麼角色？
- `requires_grad=True` 的用途是什麼？

---

## 第 4 章：Optimizer 與 AdamW — 參數如何被更新

### 4.1 這章要解決什麼問題？

有了 gradient 之後，optimizer 會決定參數如何更新。LLM 訓練常使用 AdamW，因為它能穩定處理大規模參數更新。

### 4.2 AdamW 演算法的關鍵公式

在每個時間步 $t$，對每個參數 $\theta$：

1. **更新一階動量**：

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

2. **更新二階動量**：

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

3. **偏差修正**：

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

4. **Decoupled 權重衰減與更新**：

$$\theta_{t+1} = \theta_t - \eta \cdot \lambda \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

### 4.3 💻 AdamW 自訂 step 實作

你可以在 `src/nn/optimizer.py` 中實作：

```python
state = self.state[p]
exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
step = state['step']

p.mul_(1.0 - lr * wd)

exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

bias_correction1 = 1.0 - beta1 ** step
bias_correction2 = 1.0 - beta2 ** step

step_size = lr / bias_correction1
denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

p.addcdiv_(exp_avg, denom, value=-step_size)
```

### 4.4 互動視覺化

<div id="optimizer-visualizer" class="interactive-visualizer"></div>

### 4.5 本章自我檢查

- SGD、Momentum、Adam、AdamW 差在哪？
- AdamW 的 weight decay 為什麼叫 decoupled？
- learning rate 太大或太小會發生什麼事？

---

## 第 5 章：Tokenization 與 Embedding — 文字如何變成向量

### 5.1 這章要解決什麼問題？

神經網路不能直接處理文字。LLM 的第一步是把文字切成 token，轉成 token id，再透過 embedding table 查表成向量。

### 5.2 從文字到 token id

簡化流程：

```text
文字：我 喜歡 學 AI
Tokens: ["我", "喜歡", "學", "AI"]
Token IDs: [3, 8, 12, 20]
```

真實 LLM 通常使用 BPE、SentencePiece 或其他 subword tokenizer；本書前期會先用簡化 tokenizer 幫助理解主流程。

### 5.3 Embedding Table

Embedding table 是一個可學習矩陣：

```text
embedding_table: (V, C)
```

其中：

- `V` 是 vocabulary size
- `C` 是 hidden size / embedding dimension

查表後：

```text
input_ids:  (B, T)
embedding:  (B, T, C)
```

### 5.4 💻 PyTorch 範例

```python
import torch
import torch.nn as nn

vocab_size = 100
hidden_size = 8
embedding = nn.Embedding(vocab_size, hidden_size)

input_ids = torch.tensor([[3, 8, 12, 20]])
x = embedding(input_ids)
print(x.shape)  # torch.Size([1, 4, 8])
```

### 5.5 🧩 從零手刻：SimpleTokenizer 與 SimpleEmbedding

本章對應程式碼已補在：

```text
src/tokenization/simple_tokenizer.py
src/tokenization/embedding.py
```

為了讓初學者先理解主流程，我們先使用「空白切詞」的簡化 tokenizer，而不是一開始就進入 BPE / SentencePiece。

```python
from src.tokenization import SimpleTokenizer, SimpleEmbedding

# 1. 建立簡化詞彙表
# 內建保留：<pad> = 0, <unk> = 1
tokenizer = SimpleTokenizer(tokens=["我", "喜歡", "學", "AI"])

# 2. 文字 → token ids
input_ids = tokenizer.batch_encode([
    "我 喜歡",
    "我 喜歡 學 AI",
])
print(input_ids)
# tensor([[2, 3, 0, 0],
#         [2, 3, 4, 5]])

# 3. token ids → embedding vectors
embedding = SimpleEmbedding(vocab_size=tokenizer.vocab_size, hidden_size=8)
x = embedding(input_ids)
print(x.shape)
# torch.Size([2, 4, 8])
```

### 5.6 ✅ 單元測試驗證

本章測試已補在：

```text
tests/test_token_embedding.py
```

測試涵蓋：

- 已知 token 可以正確 encode / decode
- 未知 token 會被映射到 `<unk>`
- batch encode 會 pad 成 `(B, T)` 的 `torch.long` tensor
- embedding lookup 會把 `(B, T)` 轉成 `(B, T, C)`
- embedding 輸出列會對應到 lookup table 中正確的 row

執行：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_token_embedding.py -q
```

> 如果你的 pytest 環境沒有外部 plugin 衝突，也可以直接執行 `pytest tests/test_token_embedding.py -q`。

### 5.7 互動視覺化

<div id="token-embedding-visualizer" class="interactive-visualizer"></div>

### 5.8 本章自我檢查

- token、token id、embedding vector 差在哪？
- 為什麼 `input_ids` 是整數，但 embedding 是浮點向量？
- `(B, T)` 如何變成 `(B, T, C)`？

---

## 第 6 章：Attention 直覺與 Scaled Dot-Product Attention — 模型如何決定要看哪裡

### 6.1 這章要解決什麼問題？

Attention 是 Transformer 的核心。它讓每個 token 可以根據內容決定要關注序列中的哪些 token。

### 6.2 Q / K / V 直覺

- **Query (Q)**：我現在想找什麼資訊？
- **Key (K)**：每個 token 提供什麼可被搜尋的標籤？
- **Value (V)**：每個 token 真正要被取用的內容。

核心公式：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 6.3 Scaled Dot-Product Attention 手動算一遍

假設輸入序列只有 2 個 token，每個 token 的特徵維度 $d_k = 4$：

$$Q = \begin{pmatrix} 1.0 & 0.0 & 1.0 & 0.0 \\ 0.0 & 2.0 & 0.0 & 1.0 \end{pmatrix}, \quad K = \begin{pmatrix} 1.0 & 0.0 & 1.0 & 0.0 \\ 0.0 & 1.0 & 0.0 & 1.0 \end{pmatrix}$$

1.  **計算 $QK^T$**：

$$QK^T = \begin{pmatrix} 2.0 & 0.0 \\ 0.0 & 3.0 \end{pmatrix}$$

2.  **除以縮放因子 $\sqrt{d_k}$**：

因為 $d_k = 4$，所以 $\sqrt{d_k} = 2.0$。

$$\text{Scaled Scores} = \begin{pmatrix} 1.0 & 0.0 \\ 0.0 & 1.5 \end{pmatrix}$$

3.  **套用 Softmax**：

第一列 `[1.0, 0.0]`：

$$\text{softmax}([1.0, 0.0]) \approx [0.73, 0.27]$$

代表第 1 個 token 對自己分配 73% 注意力，對第 2 個 token 分配 27% 注意力。

### 6.4 Shape 拆解

```text
Q:       (B, T, D)
K:       (B, T, D)
V:       (B, T, D)
QK^T:    (B, T, T)
weights: (B, T, T)
output:  (B, T, D)
```

### 6.5 互動視覺化

<div id="attention-visualizer" class="interactive-visualizer"></div>

### 6.6 本章自我檢查

- Q、K、V 分別代表什麼？
- 為什麼 attention weights 每列加總應該等於 1？
- 為什麼要除以 $\sqrt{d_k}$？

---

## 第 7 章：Multi-Head Attention — 用多個視角理解上下文

### 7.1 這章要解決什麼問題？

單一 attention head 只能用一種子空間觀察 token 關係。Multi-Head Attention 讓模型同時用多個視角看上下文。

### 7.2 多頭注意力 Shape 流程

```text
X:       (B, T, C)
Q/K/V:   (B, T, C)
split:   (B, H, T, D)
score:   (B, H, T, T)
context: (B, H, T, D)
merge:   (B, T, C)
output:  (B, T, C)
```

其中：

```text
D = C / H
```

### 7.3 💻 Split 與 Merge 實作

你可以在 `src/transformer/attention.py` 中使用 `view` 和 `transpose` 來處理多頭分拆：

```python
def forward(self, q, k, v, mask=None):
    batch_size, seq_len, d_model = q.shape

    q_heads = self.W_q(q).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    k_heads = self.W_k(k).view(batch_size, k.size(1), self.num_heads, self.d_k).transpose(1, 2)
    v_heads = self.W_v(v).view(batch_size, k.size(1), self.num_heads, self.d_k).transpose(1, 2)

    scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / math.sqrt(self.d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == float('-inf'), float('-inf'))
    attn_weights = torch.softmax(scores, dim=-1)
    context = torch.matmul(attn_weights, v_heads)

    context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
    return self.W_o(context)
```

### 7.4 互動視覺化

<div id="multi-head-attention-visualizer" class="interactive-visualizer"></div>

### 7.5 本章自我檢查

- Multi-head 的好處是什麼？
- 為什麼 `C` 必須能被 `H` 整除？
- split heads 與 merge heads 的 shape 如何變化？

---

## 第 8 章：Causal Mask 與 Decoder-only LLM — 為什麼不能偷看未來

### 8.1 這章要解決什麼問題？

GPT 類模型是 decoder-only autoregressive model。訓練時它要預測下一個 token，因此每個位置只能看自己與過去，不能看未來答案。

### 8.2 Next-token Prediction

若訓練資料是：

```text
我 喜歡 學 AI
```

模型任務是：

```text
輸入：我          → 預測：喜歡
輸入：我 喜歡     → 預測：學
輸入：我 喜歡 學  → 預測：AI
```

### 8.3 Causal Mask Matrix

```text
       我   喜歡   學   AI
我     ✅   ❌    ❌   ❌
喜歡   ✅   ✅    ❌   ❌
學     ✅   ✅    ✅   ❌
AI     ✅   ✅    ✅   ✅
```

在 attention score 中，被遮住的位置通常會被設成 `-inf`，softmax 後權重會變成 0。

### 8.4 💻 Causal Mask 範例

```python
import torch

T = 4
mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
scores = torch.zeros(T, T)
scores = scores.masked_fill(mask, float('-inf'))
weights = torch.softmax(scores, dim=-1)
```

### 8.5 互動視覺化

<div id="causal-mask-visualizer" class="interactive-visualizer"></div>

### 8.6 本章自我檢查

- decoder-only LLM 為什麼不能看未來 token？
- causal mask 通常套用在 attention 的哪一步？
- 被 mask 的位置 softmax 後權重是多少？

---

## 第 9 章：RMSNorm、Residual Connection 與 Pre-Norm — 讓深層模型穩定訓練

### 9.1 這章要解決什麼問題？

深層 Transformer 如果沒有 normalization 與 residual connection，訓練容易不穩定。現代 LLM 常用 RMSNorm 與 Pre-Norm 架構。

### 9.2 RMSNorm 公式

RMSNorm 相比 LayerNorm 去除了減去均值步驟，公式如下：

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}$$

### 9.3 💻 RMSNorm 實作

你可以在 `src/transformer/normalization.py` 中寫出：

```python
class CustomRMSNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return x_normed * self.gamma
```

### 9.4 Residual + Pre-Norm Block

現代 LLM block 常見形式：

```text
x = x + Attention(RMSNorm(x))
x = x + FFN(RMSNorm(x))
```

### 9.5 互動視覺化

<div id="normalization-visualizer" class="interactive-visualizer"></div>

### 9.6 本章自我檢查

- RMSNorm 與 LayerNorm 的差異是什麼？
- residual connection 為什麼能幫助深層模型？
- Pre-Norm 的資料流長什麼樣子？

---

## 第 10 章：Feed Forward Network 與 SwiGLU — Transformer 的非線性表達能力

### 10.1 這章要解決什麼問題？

Attention 負責 token 之間交換資訊，FFN 則負責對每個 token 的 hidden state 做非線性轉換。現代 LLM 常使用 SwiGLU 作為 FFN 結構。

### 10.2 SwiGLU 公式

SwiGLU 將前饋網路雙分支相乘，並以 Swish / SiLU 當作門控：

$$\text{SwiGLU}(x) = (\text{Swish}(x W_{gate}) \odot x W_{up}) W_{down}$$

### 10.3 💻 SwiGLU 實作

你可以在 `src/transformer/feed_forward.py` 中實作：

```python
class SwiGLUFFN(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_branch = F.silu(self.w_gate(x))
        up_branch = self.w_up(x)
        return self.w_down(gate_branch * up_branch)
```

### 10.4 Shape 拆解

```text
x:           (B, T, C)
gate_branch: (B, T, hidden_dim)
up_branch:   (B, T, hidden_dim)
merged:      (B, T, hidden_dim)
output:      (B, T, C)
```

### 10.5 互動視覺化

<div id="swiglu-visualizer" class="interactive-visualizer"></div>

### 10.6 本章自我檢查

- FFN 在 Transformer 中負責什麼？
- SwiGLU 的 gate branch 有什麼作用？
- 為什麼 FFN 中間維度通常會放大？

---

## 第 11 章：RoPE 旋轉位置編碼 — 讓模型知道相對位置

### 11.1 這章要解決什麼問題？

純 Attention 不知道 token 的順序。RoPE 透過旋轉 Q/K 向量，把位置資訊注入 attention 計算中。

### 11.2 RoPE 直覺

傳統位置編碼像貼上絕對頁碼；RoPE 則像給向量裝上旋轉指北針。位置越後面，向量旋轉角度越大；兩個 token 之間的相對距離反映在角度差中。

### 11.3 簡化公式

對一組二維向量 $(x_1, x_2)$，依位置 $m$ 旋轉角度 $m\theta$：

$$
\begin{pmatrix}
x_1' \\
x_2'
\end{pmatrix}
=
\begin{pmatrix}
\cos(m\theta) & -\sin(m\theta) \\
\sin(m\theta) & \cos(m\theta)
\end{pmatrix}
\begin{pmatrix}
x_1 \\
x_2
\end{pmatrix}
$$

### 11.4 Shape 拆解

RoPE 通常套用在 Q/K 上：

```text
q_heads: (B, H, T, D)
k_heads: (B, H, T, D)
```

套用 RoPE 後 shape 不變。

### 11.5 互動視覺化

<div id="rope-visualizer" class="interactive-visualizer"></div>

### 11.6 本章自我檢查

- 為什麼 Transformer 需要位置資訊？
- RoPE 為什麼套用在 Q/K 而不是 V？
- 旋轉後向量的 norm 是否應該大致保持不變？

---

## 第 12 章：KV Cache 與 LLM 推理加速 — 不重算過去的祕密

### 12.1 這章要解決什麼問題？

自迴歸解碼生成新 token 時，如果每一步都重算整段 prompt，成本會很高。KV Cache 會快取過去的 Key 與 Value，讓每一步只需處理新 token。

### 12.2 Prefill 與 Decode

```text
Prefill：一次處理整段 prompt，建立初始 K/V cache
Decode：每次只輸入新產生的 1 個 token，並把新的 K/V append 到 cache
```

### 12.3 KV Cache 維度變化

在推理的單步解碼中：

*   新輸入序列長度為 `1`，輸入維度是 `(B, 1, C)`。
*   投影後的 $K_{new}, V_{new}$ 維度是 `(B, H, 1, D)`。
*   歷史快取 $K_{cache}, V_{cache}$ 維度是 `(B, H, L_prev, D)`。
*   沿著序列維度拼接後：

$$K_{cache} \leftarrow \text{cat}([K_{cache}, K_{new}], \text{dim}=2)$$

$$V_{cache} \leftarrow \text{cat}([V_{cache}, V_{new}], \text{dim}=2)$$

新 shape：

```text
(B, H, L_prev + 1, D)
```

### 12.4 💻 KV Cache 更新虛擬碼

```python
q_new, k_new, v_new = project(x)  # k_new: (B, H, 1, D)

k_cache = torch.cat([k_cache, k_new], dim=2)
v_cache = torch.cat([v_cache, v_new], dim=2)

scores = torch.matmul(q_new, k_cache.transpose(-2, -1))
# scores: (B, H, 1, L_prev + 1)
```

### 12.5 互動視覺化

<div id="kv-cache-visualizer" class="interactive-visualizer"></div>

### 12.6 本章自我檢查

- prefill 與 decode 差在哪？
- KV Cache 快取的是 Q、K、V 中的哪兩個？
- 每次 decode 後 cache length 如何變化？

---

## 第 13 章：Mini Decoder-only LLM 組裝 — 從元件到完整模型

### 13.1 這章要解決什麼問題？

前面我們已經拆解了 LLM 的主要零件。這一章要把它們組成一個最小可運作的 Decoder-only LLM。

### 13.2 Mini LLM 資料流

```text
input_ids
 → token embedding
 → transformer block 1
 → transformer block 2
 → final RMSNorm
 → LM Head
 → logits
 → next token probability
```

### 13.3 最小模型結構

```python
class MiniDecoderOnlyLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([
            MiniTransformerBlock(config) for _ in range(config.num_layers)
        ])
        self.norm = CustomRMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
```

### 13.4 輸出 Shape

```text
input_ids: (B, T)
logits:    (B, T, V)
```

### 13.5 互動視覺化

<div id="mini-llm-visualizer" class="interactive-visualizer"></div>

### 13.6 本章自我檢查

- Mini LLM 最小需要哪些元件？
- 為什麼 LM Head 的輸出維度是 vocabulary size？
- `logits[:, -1, :]` 通常代表什麼？

---

## 第 14 章：測試驅動理解 LLM — 用 pytest 驗證你的理解

### 14.1 這章要解決什麼問題？

測試不只是工程品質工具，也可以是學習 AI 的工具。只要測試寫得好，它能幫你確認自己是否真的理解公式、shape 與數值行為。

### 14.2 測試類型

| 測試類型 | 目的 | 例子 |
|---|---|---|
| Shape test | 確認輸入輸出維度 | `assert logits.shape == (B, T, V)` |
| Numerical test | 確認數值計算正確 | `torch.allclose(custom, reference)` |
| Mask test | 確認未來 token 被遮住 | future weights 為 0 |
| Regression test | 確認修改後不破壞既有行為 | 固定 seed 比對輸出 |

### 14.3 建議測試命令

安裝依賴：

```bash
pip install -r requirements.txt
```

執行基礎測試：

```bash
pytest tests/test_foundations.py
```

執行 Transformer 測試：

```bash
pytest tests/test_transformer.py
```

執行全部測試：

```bash
pytest
```

### 14.4 本章自我檢查

- 為什麼 shape test 很適合用來學 Transformer？
- `atol` 與 `rtol` 在 numerical test 中代表什麼？
- 為什麼要用 PyTorch 官方模組當 reference？

---

## 補充 A：RNN 與 LSTM — 序列模型的歷史脈絡

### A.1 為什麼放在補充篇？

RNN / LSTM 是早期處理序列資料的重要模型。理解它們可以幫助我們看懂 Transformer 解決了什麼問題，但如果目標是學 LLM 底層架構，主線應該優先放在 Attention 與 Decoder-only Transformer。

### A.2 LSTM 門控更新公式與維度對照

LSTM 維護兩個狀態：

*   $C_t$：Cell State，長短期記憶載體。
*   $h_t$：Hidden State，當前步輸出。

在 LSTMCell 中，我們要為四個門分別計算輸入與隱藏狀態的投影：

*   忘記門 $f_t$
*   輸入門 $i_t$
*   候選門 $g_t$
*   輸出門 $o_t$

### A.3 💻 LSTMCell 實作對照

你可以在 `src/rnn/lstm_cell.py` 中實作：

```python
def forward(self, x: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    h_prev, c_prev = states

    f_t = torch.sigmoid(F.linear(x, self.W_if, self.b_if) + F.linear(h_prev, self.W_hf, self.b_hf))
    i_t = torch.sigmoid(F.linear(x, self.W_ii, self.b_ii) + F.linear(h_prev, self.W_hi, self.b_hi))
    g_t = torch.tanh(F.linear(x, self.W_ig, self.b_ig) + F.linear(h_prev, self.W_hg, self.b_hg))
    c_next = f_t * c_prev + i_t * g_t
    o_t = torch.sigmoid(F.linear(x, self.W_io, self.b_io) + F.linear(h_prev, self.W_ho, self.b_ho))
    h_next = o_t * torch.tanh(c_next)

    return h_next, c_next
```

### A.4 Transformer 相比 RNN / LSTM 的優勢

- Attention 可以直接看任意位置，較容易捕捉長距離依賴。
- Transformer 訓練時更容易平行化。
- LLM 的主流架構多採用 Decoder-only Transformer。

---

## 補充 B：CNN 與局部特徵 — 卷積如何看見空間結構

### B.1 為什麼放在補充篇？

CNN 是理解局部特徵、權重共享與感受野的重要模型。它不是 Decoder-only LLM 的核心主線，但能補充你對深度學習架構的整體理解。

### B.2 2D 卷積滑動視窗計算公式

當使用核心大小為 $K \times K$、步長為 $S$、填充為 $P$ 的卷積核處理輸入時，輸出高度計算公式為：

$$H_{out} = \left\lfloor \frac{H_{in} - K_h + 2P_h}{S_h} \right\rfloor + 1$$

### B.3 🧐 手動算一遍：輸出維度計算

設：

*   輸入高度 $H_{in} = 5$
*   Kernel 高度 $K_h = 3$
*   Padding $P_h = 1$
*   Stride $S_h = 2$

帶入公式：

$$H_{out} = \left\lfloor \frac{5 - 3 + 2 \times 1}{2} \right\rfloor + 1 = 3$$

代表卷積層輸出特徵圖高度為 $3$。

### B.4 互動視覺化

<div id="cnn-visualizer" class="interactive-visualizer"></div>

---

## 後續撰寫原則

每章後續擴寫時，建議維持以下固定格式：

```markdown
## 第 N 章：主題名稱

### N.1 這章要解決什麼問題？
### N.2 白話直覺
### N.3 數學公式
### N.4 手算一個小例子
### N.5 Tensor shape 拆解
### N.6 PyTorch 實作
### N.7 從零手刻版本
### N.8 單元測試驗證
### N.9 互動視覺化
### N.10 本章自我檢查
```

綠燈（Pass）代表你的自製組件在數學與 shape 上通過驗證。祝你手寫 LLM 順利！
