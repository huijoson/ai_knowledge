# 《從零手刻 LLM：Transformer 核心原理互動電子書》課程規劃

> 本文件是 `ai_knowledge` 專案的自學型互動電子書課程藍圖。目標是幫助想理解 LLM 底層架構的人，從張量、線性層、反向傳播一路走到 Transformer、RoPE、KV Cache 與 Mini LLM 實作。

---

## 1. 課程定位

### 課程名稱

**《從零手刻 LLM：Transformer 核心原理互動電子書》**

### 一句話定位

用「白話直覺、數學公式、手算範例、互動視覺化、PyTorch 對照、從零實作、測試驗證」的方式，系統性理解 LLM 的底層架構。

### 核心精神

```text
白話直覺 → 數學原理 → 手算範例 → Tensor Shape → 互動視覺化 → PyTorch 對照 → 從零實作 → 單元測試驗證
```

這套課程不是 API 使用教學，也不是只教如何呼叫現成模型，而是要理解：

- 文字如何變成 token 與 embedding
- Transformer block 裡每個元件在做什麼
- Attention 如何計算「該看哪裡」
- Multi-Head Attention 為什麼需要多個 head
- Decoder-only LLM 為什麼需要 causal mask
- RMSNorm、Residual、SwiGLU 如何讓模型更穩定與更有表達力
- RoPE 如何讓模型理解位置與相對距離
- KV Cache 如何加速 LLM 推理
- 如何把這些元件組成一個 Mini LLM

---

## 2. 目標讀者

### 主要讀者

想學習 **LLM 底層架構與 AI 理論基礎** 的自學者。

### 適合背景

讀者不需要已經懂 Transformer，但建議具備：

- 基礎 Python 程式能力
- 看得懂簡單矩陣與向量概念
- 願意一邊看理論、一邊手算、一邊跑程式

### 不以這些讀者為主要對象

- 只想快速使用 ChatGPT API 的人
- 只想做 prompt engineering 的人
- 不想碰任何數學、程式或 tensor shape 的人

---

## 3. 學習成果

完成本電子書後，學習者應該能夠：

1. 解釋 LLM 的整體資料流：文字 → token → embedding → Transformer blocks → logits → next token。
2. 看懂常見 LLM tensor shape，例如 `(B, T, C)`、`(B, H, T, D)`。
3. 手算小型 Linear Layer、Attention、Softmax 與 KV Cache 範例。
4. 從零實作 Linear、Attention、RMSNorm、SwiGLU、RoPE、KV Cache 等核心元件。
5. 對照 PyTorch 官方實作，確認自訂模組的數值與 shape 正確。
6. 理解 Decoder-only LLM 的 next-token prediction 訓練邏輯。
7. 組裝一個可 forward 的 Mini LLM。
8. 使用 pytest 寫 shape test、numerical equivalence test 與 regression test。

---

## 4. 建議課程主線

本課程建議採取 **LLM-first** 的學習順序，而不是傳統深度學習課程的 CNN/RNN-first。

```text
Tensor / Shape
 → Linear Layer
 → Autograd
 → Optimizer
 → Tokenization / Embedding
 → Attention
 → Multi-Head Attention
 → Causal Mask
 → RMSNorm / Residual
 → SwiGLU FFN
 → RoPE
 → KV Cache
 → Mini Decoder-only LLM
 → 測試驅動理解 LLM
```

RNN 與 CNN 建議放在補充篇，作為理解序列模型與特徵提取歷史脈絡的輔助章節。

---

## 5. 課程章節總覽

| 章節 | 主題 | 角色 |
|---|---|---|
| 第 0 章 | LLM 底層架構學習地圖 | 建立全局視野 |
| 第 1 章 | Tensor、矩陣與 Shape 直覺 | 建立張量底座 |
| 第 2 章 | Linear Layer 與基本神經網路運算 | 理解投影與維度轉換 |
| 第 3 章 | Autograd 與反向傳播 | 理解模型如何學習 |
| 第 4 章 | Optimizer 與 AdamW | 理解參數如何更新 |
| 第 5 章 | Tokenization 與 Embedding | 文字進入模型的入口 |
| 第 6 章 | Attention 直覺與 Scaled Dot-Product Attention | Transformer 核心 |
| 第 7 章 | Multi-Head Attention | 多視角注意力 |
| 第 8 章 | Causal Mask 與 Decoder-only LLM | GPT 類模型核心訓練邏輯 |
| 第 9 章 | RMSNorm、Residual Connection 與 Pre-Norm | 訓練穩定性 |
| 第 10 章 | Feed Forward Network 與 SwiGLU | Transformer 的非線性表達能力 |
| 第 11 章 | RoPE 旋轉位置編碼 | 位置與相對距離 |
| 第 12 章 | KV Cache 與推理加速 | LLM inference 關鍵優化 |
| 第 13 章 | Mini Decoder-only LLM 組裝 | 整合所有元件 |
| 第 14 章 | 測試驅動理解 LLM | 用測試驗證理解 |
| 補充 A | RNN 與 LSTM | 序列模型歷史脈絡 |
| 補充 B | CNN 與局部特徵 | 局部感受野與權重共享 |

---

## 6. 每章詳細規劃

## 第 0 章：LLM 底層架構學習地圖

### 學習目標

建立整門課的地圖，知道 LLM 是由哪些元件組成，以及後續章節會如何逐步拆解。

### 核心問題

- LLM 到底是一個什麼樣的模型？
- 為什麼 LLM 可以被看成 next-token prediction machine？
- 一段文字如何一路變成下一個 token 的機率？

### 內容重點

- AI、Machine Learning、Deep Learning、LLM 的關係
- LLM 的資料流：

```text
文字 → tokenizer → token ids → embedding → Transformer blocks → logits → softmax → next token
```

- Transformer block 的主要元件：
  - Attention
  - Normalization
  - Feed Forward Network
  - Residual Connection
- Decoder-only LLM 的基本概念

### 互動元件

**LLM 資料流動畫**

使用者輸入一句話，畫面顯示：

```text
文字 → token → token id → embedding vector → block → logits → 機率最高的下一個 token
```

### 實作任務

- 建立一個簡化版文字資料流示意，不需要真正訓練模型。
- 用固定 vocabulary 展示 token id 與 embedding lookup。

### 測試設計

- 測試輸入文字能被轉成 token list。
- 測試 token id shape 為 `(T,)` 或 `(B, T)`。
- 測試 embedding lookup 後 shape 為 `(B, T, C)`。

---

## 第 1 章：Tensor、矩陣與 Shape 直覺

### 學習目標

讓學習者理解 LLM 中最常見的 tensor shape，尤其是 `(B, T, C)`。

### 核心問題

- scalar、vector、matrix、tensor 差在哪？
- batch size、sequence length、hidden size 是什麼？
- 為什麼 Transformer 裡一直出現 shape 轉換？

### 內容重點

- Scalar：單一數字
- Vector：一列數字
- Matrix：二維表格
- Tensor：多維資料容器
- LLM 常見維度：

```text
B = batch size
T = sequence length / token count
C = hidden size / embedding dimension
H = number of attention heads
D = head dimension
V = vocabulary size
```

- 常見 shape：

```text
input_ids:      (B, T)
embedding:      (B, T, C)
attention score:(B, H, T, T)
logits:         (B, T, V)
```

### 互動元件

**Tensor Shape Visualizer**

可調整：

- B
- T
- C
- H

畫面即時顯示資料結構如何變化。

### 實作任務

- 建立 `shape_examples.py` 展示常見 LLM tensor shape。
- 用小型 tensor 印出 shape 與簡單解釋。

### 測試設計

- 測試 reshape 後維度正確。
- 測試 split heads 後 shape 正確。
- 測試 merge heads 後回到 `(B, T, C)`。

---

## 第 2 章：Linear Layer 與基本神經網路運算

### 學習目標

理解 Linear Layer 如何改變特徵維度，以及它在 Transformer 中大量出現的原因。

### 核心問題

- `Y = XW^T + b` 到底在做什麼？
- Linear Layer 如何把 hidden size 從 C 轉成另一個維度？
- Q、K、V projection 本質上是不是 Linear Layer？

### 內容重點

- Linear Layer 公式：

```text
Y = XW^T + b
```

- 權重矩陣 shape：

```text
X: (B, T, in_features)
W: (out_features, in_features)
b: (out_features)
Y: (B, T, out_features)
```

- Transformer 中 Linear 的用途：
  - Q projection
  - K projection
  - V projection
  - Output projection
  - FFN up projection
  - FFN gate projection
  - FFN down projection
  - LM Head

### 互動元件

**Matrix Multiplication Visualizer**

顯示每個輸出值如何由輸入 vector 與權重 row 相乘加總而來。

### 實作任務

- 從零實作 `CustomLinear`。
- 與 `torch.nn.Linear` 對照。
- 寫一個手算小例子。

### 測試設計

- 測試輸出 shape。
- 測試在同樣 weight / bias 下，`CustomLinear` 與 `torch.nn.Linear` 數值一致。

---

## 第 3 章：Autograd 與反向傳播

### 學習目標

理解模型如何透過 loss 與 gradient 更新參數。

### 核心問題

- forward pass 與 backward pass 是什麼？
- chain rule 如何讓模型知道每個參數該怎麼改？
- PyTorch 的 autograd 幫我們做了什麼？

### 內容重點

- Computation Graph
- Chain Rule
- Gradient
- `requires_grad=True`
- `.backward()`
- `.grad`

### 互動元件

**Autograd Computation Graph Visualizer**

範例：

```text
x, y, w → u = x + y → z = u * w
```

互動流程：

1. 調整 x、y、w。
2. 點擊「前向傳播」。
3. 點擊「反向傳播」。
4. 顯示每個節點的 value 與 gradient。

### 實作任務

- 手算一個簡單函數的梯度。
- 用 PyTorch autograd 驗證。

### 測試設計

- 測試手算 gradient 與 PyTorch gradient 一致。

---

## 第 4 章：Optimizer 與 AdamW

### 學習目標

理解訓練時參數如何根據 gradient 被 optimizer 更新，以及為什麼 LLM 常用 AdamW。

### 核心問題

- SGD、Momentum、Adam、AdamW 差在哪？
- learning rate 是什麼？
- weight decay 為什麼要 decoupled？

### 內容重點

- Gradient Descent
- Momentum
- Adam 的一階與二階動量
- Bias correction
- Weight Decay
- AdamW

### 互動元件

**Optimizer 小球下山視覺化**

比較：

- SGD
- Momentum
- AdamW

### 實作任務

- 從零實作簡化版 AdamW step。
- 與 PyTorch optimizer 做小型對照。

### 測試設計

- 測試單步更新結果是否與參考公式一致。
- 測試 weight decay 是否正確影響參數。

---

## 第 5 章：Tokenization 與 Embedding

### 學習目標

理解文字如何變成模型可以計算的向量。

### 核心問題

- 為什麼文字不能直接丟進神經網路？
- token、token id、vocabulary 是什麼？
- embedding table 是什麼？

### 內容重點

- Tokenization
- Vocabulary
- Token ID
- Embedding Table
- Lookup operation
- Shape：

```text
input_ids: (B, T)
embedding_table: (V, C)
output_embeddings: (B, T, C)
```

### 互動元件

**Token → ID → Embedding Visualizer**

使用者輸入：

```text
我 喜歡 學 AI
```

畫面顯示：

```text
tokens → token ids → embedding vectors
```

### 實作任務

- 建立簡化 tokenizer。
- 建立簡化 embedding table。
- 實作 embedding lookup。

### 測試設計

- 測試 token 對應 id 正確。
- 測試 unknown token 處理。
- 測試 embedding output shape。

---

## 第 6 章：Attention 直覺與 Scaled Dot-Product Attention

### 學習目標

理解 Transformer 的核心運算：Attention。

### 核心問題

- 模型如何知道每個 token 應該關注哪些 token？
- Q、K、V 分別代表什麼？
- `QK^T / sqrt(d_k)` 為什麼要除以 `sqrt(d_k)`？
- Softmax 在 Attention 裡做什麼？

### 內容重點

- Query / Key / Value 直覺
- Attention Score
- Scaling
- Softmax
- Weighted Sum
- 公式：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

### 互動元件

**Attention Heatmap Visualizer**

輸入一句話，例如：

```text
貓 坐 在 墊子 上
```

使用者點選某個 token，畫面顯示它對其他 token 的注意力權重。

### 實作任務

- 手算一個小型 Q/K/V 範例。
- 從零實作 scaled dot-product attention。
- 顯示 attention weights。

### 測試設計

- 測試 attention weights 每列總和為 1。
- 測試輸出 shape。
- 測試小型手算範例數值一致。

---

## 第 7 章：Multi-Head Attention

### 學習目標

理解多頭注意力如何讓模型從多個子空間觀察 token 關係。

### 核心問題

- 為什麼不只用一個 attention head？
- split heads 與 merge heads 是什麼？
- 每個 head 的 shape 如何變化？

### 內容重點

- Single-head vs Multi-head
- Q/K/V projection
- Split heads
- Head dimension
- Concatenate heads
- Output projection

### Shape 流程

```text
X:       (B, T, C)
Q/K/V:   (B, T, C)
split:   (B, H, T, D)
score:   (B, H, T, T)
context: (B, H, T, D)
merge:   (B, T, C)
output:  (B, T, C)
```

### 互動元件

**Multi-Head Attention Heatmap**

同一句話顯示不同 head 的注意力熱圖。

### 實作任務

- 實作 split heads。
- 實作 merge heads。
- 實作完整 Multi-Head Attention。

### 測試設計

- 測試 split / merge 後資料 shape 正確。
- 測試 MHA output shape。
- 測試 causal mask 可正確套用。

---

## 第 8 章：Causal Mask 與 Decoder-only LLM

### 學習目標

理解 GPT 類模型為什麼只能看過去 token，不能偷看未來 token。

### 核心問題

- Encoder、Decoder、Decoder-only 差在哪？
- Next-token prediction 是什麼？
- causal mask 如何防止模型偷看答案？

### 內容重點

- Autoregressive generation
- Decoder-only Transformer
- Next-token prediction
- Causal attention
- 上三角 mask

### 互動元件

**Causal Mask Matrix Visualizer**

顯示：

```text
       我   喜歡   學   AI
我     ✅   ❌    ❌   ❌
喜歡   ✅   ✅    ❌   ❌
學     ✅   ✅    ✅   ❌
AI     ✅   ✅    ✅   ✅
```

### 實作任務

- 產生 causal mask。
- 將 causal mask 套用到 attention score。
- 驗證未來位置的 attention weight 為 0。

### 測試設計

- 測試 mask shape 為 `(T, T)` 或可 broadcast 到 `(B, H, T, T)`。
- 測試未來 token 權重為 0。
- 測試 masked softmax 不產生 NaN。

---

## 第 9 章：RMSNorm、Residual Connection 與 Pre-Norm

### 學習目標

理解現代 LLM block 中 normalization 與 residual connection 的重要性。

### 核心問題

- 為什麼需要 normalization？
- RMSNorm 與 LayerNorm 差在哪？
- residual connection 如何幫助深層模型訓練？
- Pre-Norm 架構為什麼常見？

### 內容重點

- LayerNorm vs RMSNorm
- RMS 計算
- Scale parameter
- Residual Connection
- Pre-Norm Transformer Block：

```text
x = x + Attention(RMSNorm(x))
x = x + FFN(RMSNorm(x))
```

### 互動元件

**Normalization 數值分佈視覺化**

顯示 normalization 前後的向量尺度變化。

### 實作任務

- 從零實作 RMSNorm。
- 加入 residual connection 範例。

### 測試設計

- 測試 RMSNorm output shape 不變。
- 測試 RMSNorm 與參考實作數值一致。
- 測試 residual connection output shape 正確。

---

## 第 10 章：Feed Forward Network 與 SwiGLU

### 學習目標

理解 Transformer block 中 FFN 的角色，以及現代 LLM 為什麼常用 SwiGLU。

### 核心問題

- Attention 之外，FFN 在 Transformer 中做什麼？
- 為什麼 FFN 中間維度通常會放大？
- SwiGLU 的 gate branch 與 up branch 是什麼？

### 內容重點

- FFN
- Up projection
- Down projection
- GELU / SiLU
- GLU
- SwiGLU 公式：

```text
SwiGLU(x) = SiLU(xW_gate) ⊙ (xW_up)
output = SwiGLU(x) W_down
```

### 互動元件

**SwiGLU Gate Visualizer**

顯示：

```text
x → gate branch → SiLU → element-wise multiply
x → up branch   ────────────────┘
                         → down projection
```

### 實作任務

- 實作簡單 FFN。
- 實作 SwiGLU FFN。
- 對照不同 activation 的輸出變化。

### 測試設計

- 測試 output shape。
- 測試 gate branch 與 up branch shape。
- 測試小型手算範例。

---

## 第 11 章：RoPE 旋轉位置編碼

### 學習目標

理解 LLM 如何知道 token 的位置，以及 RoPE 如何用旋轉表示相對位置。

### 核心問題

- Transformer 如果沒有位置資訊會發生什麼？
- Absolute positional embedding 與 RoPE 差在哪？
- RoPE 為什麼可以保留相對距離資訊？

### 內容重點

- Positional Encoding
- Absolute position
- Sinusoidal position encoding
- RoPE
- 向量旋轉
- 相對位置與角度差

### 互動元件

**RoPE Rotation Animator**

使用者拖動 position，畫面顯示向量旋轉角度：

```text
position 0 → 角度 0θ
position 1 → 角度 1θ
position 2 → 角度 2θ
position 3 → 角度 3θ
```

### 實作任務

- 產生 cos / sin cache。
- 對 Q/K 套用 RoPE。
- 顯示不同 position 的向量變化。

### 測試設計

- 測試 RoPE 不改變 shape。
- 測試 position 0 結果合理。
- 測試旋轉後向量 norm 近似不變。

---

## 第 12 章：KV Cache 與推理加速

### 學習目標

理解 LLM 生成文字時，KV Cache 如何避免重複計算，提升推理效率。

### 核心問題

- prefill 與 decode 差在哪？
- 為什麼沒有 KV Cache 會一直重算前文？
- K cache 與 V cache 的 shape 如何隨生成長度變化？

### 內容重點

- Autoregressive decoding
- Prefill stage
- Decode stage
- Past key values
- Cache append
- Cache shape：

```text
K cache: (B, H, past_len, D)
V cache: (B, H, past_len, D)
```

### 互動元件

**KV Cache Timeline Visualizer**

逐 token 顯示：

```text
Prompt: 我 喜歡
生成: 學
生成: AI
生成: 的
生成: 原理
```

並同步顯示 K/V cache 長度成長。

### 實作任務

- 實作不使用 cache 的 attention 推理。
- 實作使用 cache 的 attention 推理。
- 比較兩者重算的 token 數量。

### 測試設計

- 測試 cache length 每次 decode 增加 1。
- 測試 cache shape 正確。
- 測試使用 cache 與不使用 cache 的 logits 在小型範例中一致或近似一致。

---

## 第 13 章：Mini Decoder-only LLM 組裝

### 學習目標

把前面所有元件整合成一個簡化版 Decoder-only LLM。

### 核心問題

- 一個 GPT-like model 最小需要哪些元件？
- input ids 如何一路變成 logits？
- logits 如何對應 vocabulary 中每個 token 的分數？

### 內容重點

- Token Embedding
- Transformer Block
- RMSNorm
- Multi-Head Attention
- RoPE
- SwiGLU FFN
- LM Head
- Logits
- Softmax
- Next-token prediction

### 模型資料流

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

### 互動元件

**Mini LLM Forward Pass Visualizer**

顯示一筆輸入如何經過每個層，並標示每一步 shape。

### 實作任務

- 建立 `MiniLLMConfig`。
- 實作 `MiniTransformerBlock`。
- 實作 `MiniDecoderOnlyLLM`。
- 跑一次 forward。

### 測試設計

- 測試 `logits.shape == (B, T, vocab_size)`。
- 測試模型可處理不同 sequence length。
- 測試 causal mask 在 Mini LLM 中生效。

---

## 第 14 章：測試驅動理解 LLM

### 學習目標

把測試變成學習 AI 的工具，用測試確認自己真的理解 shape、公式與行為。

### 核心問題

- 為什麼學 AI 也需要寫測試？
- shape test、numerical test、regression test 各自驗證什麼？
- 如何用 PyTorch 官方模組作為參考答案？

### 內容重點

- Test-driven learning
- Shape tests
- Numerical equivalence tests
- Reference implementation tests
- Regression tests
- Tolerance：`atol` / `rtol`

### 互動元件

**Test Result Explainer**

顯示測試名稱、測試目的、輸入 shape、預期輸出與 pass/fail 原因。

### 實作任務

- 為每個核心模組建立至少一個 shape test。
- 為 Linear、Attention、RMSNorm、RoPE 建立 numerical test。
- 建立 Mini LLM forward test。

### 測試設計

- 整合所有測試：

```bash
pytest
```

---

## 補充 A：RNN 與 LSTM

### 定位

作為 Transformer 出現前的序列模型背景知識。

### 重點

- Hidden state
- Sequential processing
- 梯度消失
- LSTM gates
- 為什麼 Transformer 更適合長距離依賴與平行運算

### 互動元件

**序列記憶流動畫**

顯示 token 如何一個接一個更新 hidden state。

---

## 補充 B：CNN 與局部特徵

### 定位

作為理解局部感受野、權重共享與特徵提取的補充。

### 重點

- Kernel
- Stride
- Padding
- Feature map
- Local receptive field
- CNN 與 Attention 的差異

### 互動元件

**2D Convolution Sliding Kernel Visualizer**

顯示 kernel 在圖片上滑動並產生 feature map。

---

## 7. 每章固定模板

建議每章使用一致格式，降低自學負擔：

```markdown
# 第 N 章：主題名稱

## 1. 這章要解決什麼問題？

## 2. 白話直覺

## 3. 數學公式

## 4. 手算一個小例子

## 5. Tensor shape 拆解

## 6. PyTorch 實作

## 7. 從零手刻版本

## 8. 單元測試驗證

## 9. 互動視覺化

## 10. 本章總結

## 11. 自我檢查問題
```

---

## 8. 互動網站功能規劃

### 8.1 必做互動元件

| 元件 | 用途 | 對應章節 |
|---|---|---|
| LLM 資料流動畫 | 建立全局架構 | 第 0 章 |
| Tensor Shape Visualizer | 理解 shape | 第 1、7、12、13 章 |
| Matrix Multiplication Visualizer | 理解 Linear / QK^T | 第 2、6 章 |
| Autograd Graph Visualizer | 理解反向傳播 | 第 3 章 |
| Attention Heatmap | 理解注意力權重 | 第 6、7 章 |
| Causal Mask Matrix | 理解不能偷看未來 | 第 8 章 |
| Normalization Visualizer | 理解 RMSNorm | 第 9 章 |
| SwiGLU Gate Visualizer | 理解 FFN gate | 第 10 章 |
| RoPE Rotation Animator | 理解位置旋轉 | 第 11 章 |
| KV Cache Timeline | 理解推理加速 | 第 12 章 |
| Mini LLM Forward Visualizer | 串起全模型 | 第 13 章 |
| Test Result Explainer | 理解測試目的 | 第 14 章 |

### 8.2 互動元件設計原則

- 每個互動元件只教一件事。
- 數值要小，方便手算。
- 每個元件都要顯示 tensor shape。
- 優先使用固定範例，避免一開始就做過度複雜的泛用工具。
- 每個互動元件旁邊應放對應公式與 PyTorch snippet。

---

## 9. 建議檔案結構演進

目前專案已有：

```text
AI_Knowledge_Book.md
index.html
specs/
src/
tests/
```

建議後續演進成：

```text
ai_knowledge/
├── AI_Knowledge_Book.md
├── COURSE_PLAN.md
├── README.md
├── index.html
├── specs/
│   ├── 00_foundations.md
│   ├── 01_token_embedding.md
│   ├── 02_attention.md
│   ├── 03_transformer_block.md
│   ├── 04_kv_cache.md
│   └── 05_mini_llm.md
├── src/
│   ├── nn/
│   │   ├── linear.py
│   │   ├── activation.py
│   │   └── optimizer.py
│   ├── transformer/
│   │   ├── attention.py
│   │   ├── feed_forward.py
│   │   ├── normalization.py
│   │   ├── positional_encoding.py
│   │   └── model.py
│   ├── tokenization/
│   │   └── simple_tokenizer.py
│   └── visualizers/
│       └── examples.py
├── tests/
│   ├── test_foundations.py
│   ├── test_token_embedding.py
│   ├── test_attention.py
│   ├── test_transformer.py
│   ├── test_kv_cache.py
│   └── test_mini_llm.py
└── web/
    ├── components/
    ├── visualizers/
    └── styles/
```

> 注意：這是建議演進方向，不一定要一次重構。第一階段可以先維持單一 `index.html`，等互動元件變多後再拆分。

---

## 10. 建議開發順序

### Phase 1：課程骨架整理

目標：先把電子書結構改成 LLM-first。

1. 調整 `AI_Knowledge_Book.md` 章節順序。
2. 補上第 0 章 LLM 全局地圖。
3. 將 RNN / CNN 移到補充篇或歷史背景篇。
4. 每章統一加入：學習目標、核心問題、shape、測試。

### Phase 2：補齊核心文字教材

目標：讓內容可以完整自學。

1. 補 Tokenization / Embedding 章。
2. 擴充 Attention 手算範例。
3. 擴充 Causal Mask 章。
4. 擴充 RMSNorm / Residual / Pre-Norm 章。
5. 擴充 RoPE 與 KV Cache 章。
6. 新增 Mini LLM 組裝章。

### Phase 3：Python from-scratch 實作

目標：每個理論章都有對應程式碼。

1. 實作 simple tokenizer。
2. 實作 embedding lookup 範例。
3. 完善 Linear / Activation / Optimizer。
4. 完善 Attention / MHA / Causal Mask。
5. 完善 RMSNorm / SwiGLU / RoPE。
6. 實作 KV Cache。
7. 實作 Mini Decoder-only LLM。

### Phase 4：測試驗證

目標：每章至少有可執行測試。

1. 補 shape tests。
2. 補 numerical equivalence tests。
3. 補 PyTorch reference tests。
4. 補 Mini LLM forward tests。
5. 確保 `pytest` 可完整通過。

### Phase 5：互動網站升級

目標：把電子書變成真正的互動學習網站。

1. 加入 LLM 資料流動畫。
2. 加入 Tensor Shape Visualizer。
3. 加入 Matrix Multiplication Visualizer。
4. 加入 Attention Heatmap。
5. 加入 Causal Mask Matrix。
6. 加入 RoPE Rotation Animator。
7. 加入 KV Cache Timeline。
8. 加入 Mini LLM Forward Visualizer。

---

## 11. 每章驗證標準

每章完成時，至少要符合：

- [ ] 有白話解釋
- [ ] 有核心公式
- [ ] 有手算範例
- [ ] 有 tensor shape 拆解
- [ ] 有 PyTorch snippet
- [ ] 有 from-scratch 實作或清楚指出尚未實作
- [ ] 有 pytest 測試
- [ ] 有互動視覺化設計或已實作互動元件
- [ ] 有本章總結
- [ ] 有自我檢查問題

---

## 12. 風險與取捨

### 風險 1：內容過多，容易變成大而全但難完成

建議先完成 LLM 主線，不要一開始就追求涵蓋所有 AI 領域。

優先順序：

```text
Attention / MHA / Causal Mask / RMSNorm / SwiGLU / RoPE / KV Cache / Mini LLM
```

### 風險 2：互動元件太多，開發成本高

建議第一版每個互動元件都做最小可用版本。

例如 Attention Heatmap 第一版只支援固定句子與固定權重，不需要一開始就接真模型。

### 風險 3：數學太硬，降低自學動力

每個公式前都要先有白話直覺，每個公式後都要有小數字手算範例。

### 風險 4：程式碼太工程化，反而看不懂原理

from-scratch 實作應保持短小、清楚、教學導向，不追求高效能。

---

## 13. 開放問題

後續需要決定：

1. 電子書是否維持單一 `AI_Knowledge_Book.md`，還是拆成每章獨立 markdown？
2. `index.html` 是否繼續維持單檔，還是改成前端專案架構？
3. 互動元件要用純 JavaScript、React，還是維持無 build 的簡單形式？
4. 是否要加入 HuggingFace tokenizer 作為進階補充？
5. 是否要加入真正小語料訓練 Mini LLM，還是只做到 forward / inference 示意？
6. 是否要提供每章練習題與解答？

---

## 14. 第一版最小可行成果（MVP）

若要先做出一個可用的自學電子書 MVP，建議範圍如下：

### 必要章節

1. 第 0 章：LLM 學習地圖
2. 第 1 章：Tensor / Shape
3. 第 2 章：Linear Layer
4. 第 5 章：Tokenization / Embedding
5. 第 6 章：Attention
6. 第 7 章：Multi-Head Attention
7. 第 8 章：Causal Mask
8. 第 11 章：RoPE
9. 第 12 章：KV Cache
10. 第 13 章：Mini LLM

### 必要互動元件

1. LLM 資料流動畫
2. Tensor Shape Visualizer
3. Attention Heatmap
4. Causal Mask Matrix
5. RoPE Rotation Animator
6. KV Cache Timeline

### 必要測試

1. Linear shape / numerical test
2. Attention shape / weights sum test
3. Causal mask test
4. RoPE shape / norm preservation test
5. KV cache length growth test
6. Mini LLM logits shape test

---

## 15. 建議下一步

下一步建議先做：

1. 將 `AI_Knowledge_Book.md` 重整為 LLM-first 章節順序。
2. 新增第 0 章「LLM 底層架構學習地圖」。
3. 新增第 5 章「Tokenization 與 Embedding」。
4. 補強 Attention / MHA / Causal Mask / RoPE / KV Cache 的手算範例與 shape 拆解。
5. 再逐步改 `index.html` 的目錄與互動元件。

建議第一個實作任務：

> **先重寫 `AI_Knowledge_Book.md` 的目錄與章節骨架，讓整本書從一開始就朝 LLM 主線前進。**
