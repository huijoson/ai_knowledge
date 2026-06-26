# 從零開始：深入理解深度學習與大語言模型核心架構 (手把手實作指南)

歡迎閱讀這本書！本書專為希望透過「動手實作（From Scratch）」來真正理解現代人工智慧核心模型（RNN, CNN, Self-Attention, Transformer, LLM）的開發者與研究者所設計。

在本書中，我們不只是學習理論，而是會結合 **OpenSpec (規格驅動)** 與 **Superpowers (測試驅動)** 的工程思維，將每個數學公式與結構設計轉化為實際運行的 Python & PyTorch 程式碼。

---

## 📖 目錄
1. [第一章：循環神經網路 (RNN & LSTM) — 時間與序列的建模](#第一章循環神經網路-rnn--lstm--時間與序列的建模)
2. [第二章：機器視覺的基石 (CNN) — 空間與特徵的提取](#第二章機器視覺的基石-cnn--空間與特徵的提取)
3. [第三章：注意力機制 (Attention) — 從 MHA 到大模型的 GQA 與 MQA](#第三章注意力機制-attention--從-mha-到大模型的-gqa-與-mqa)
4. [第四章：現代大語言模型 (LLM) 的穩定與加速技術 — RMSNorm, SwiGLU 與 RoPE](#第四章現代大語言模型-llm-的穩定與加速技術--rmsnorm-swiglu-與-rope)
5. [第五章：經典 Transformer 與 Decoder-Only LLM 的組裝](#第五章經典-transformer-與-decoder-only-llm-的組裝)
6. [附錄：如何使用本專案進行自我驗證 (測試驅動學習)](#附錄如何使用本專案進行自我驗證-測試驅動學習)

---

## 第一章：循環神經網路 (RNN & LSTM) — 時間與序列的建模

在處理自然語言、時間序列或語音等資料時，資料點之間存在強烈的**時間前後順序關係**。傳統的密集連接網路（Dense Layer）假設輸入之間是獨立的，無法有效處理這類序列資料。這就引入了**循環神經網路（Recurrent Neural Networks, RNN）**。

### 1.1 傳統 RNN Cell 的數學與物理意義

RNN 的核心思想是維護一個**隱藏狀態（Hidden State, $h_t$）**，這個狀態就像是網路的「短期記憶」，會在時間軸上不斷傳遞。

#### 數學公式
對於每一個時間步 $t$，網路接收當前輸入 $x_t$ 以及上一個時間步的隱藏狀態 $h_{t-1}$，並計算出新的隱藏狀態 $h_t$：

$$h_t = \tanh(x_t W_{ih}^T + b_{ih} + h_{t-1} W_{hh}^T + b_{hh})$$

*   $x_t \in \mathbb{R}^{B \times D_{in}}$：當前步輸入（$B$ 是 Batch Size，$D_{in}$ 是輸入特徵維度）。
*   $h_{t-1} \in \mathbb{R}^{B \times D_{hid}}$：前一步的隱藏記憶。
*   $W_{ih} \in \mathbb{R}^{D_{hid} \times D_{in}}$：輸入到隱藏層的權重矩陣。
*   $W_{hh} \in \mathbb{R}^{D_{hid} \times D_{hid}}$：隱藏層到隱藏層的循環權重矩陣。
*   $\tanh$：激活函數，將輸出限制在 $[-1, 1]$ 之間，防止數值無限發散。

#### 為什麼傳統 RNN 會遺忘？(梯度消失與爆炸)
當我們對 RNN 進行時間反向傳播（BPTT）時，梯度需要穿過多個時間步：
$$\frac{\partial h_T}{\partial h_0} = \prod_{t=1}^T \frac{\partial h_t}{\partial h_{t-1}}$$
其中 $\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}(1 - h_t^2) W_{hh}$。
*   如果 $W_{hh}$ 的特徵值小於 1，隨著時間步 $T$ 變長，梯度會呈指數級衰減至 0（**梯度消失**），導致網路無法記得很久以前的資訊。
*   反之，若特徵值大於 1，梯度會呈指數級增長（**梯度爆炸**）。

---

### 1.2 LSTM：引入門控（Gates）機制

為了為了解決傳統 RNN 的短期記憶限制，**長短期記憶網路（Long Short-Term Memory, LSTM）** 於 1997 年被提出。它引入了**細胞狀態（Cell State, $C_t$）**，這是一條極具革命性的資訊高速公路，梯度可以在上面幾乎無損地傳播。

LSTM 的關鍵在於**門（Gates）**，門就像是閥門，由一個 Sigmoid 神經網路層和一個按元素相乘（Hadamard Product）的操作組成，用來決定有多少資訊可以通過。

```
                    Cell State (C_t)
   C_{t-1} ───────────────────⊕─────────────────── C_t
                             ▲
                             │ (i_t * g_t)
              ┌───┐        ┌─┴─┐        ┌───┐
   h_{t-1} ──►│ f ├──────► │ i ├──────► │ o ├───► h_t
              └───┘        └───┘        └───┘
              Forget       Input        Output
              Gate         Gate         Gate
```

#### LSTM 的六個核心公式
對於當前輸入 $x_t$ 和前一步隱藏狀態 $h_{t-1}$：

1.  **遺忘門 ($f_t$)**：決定要丟棄多少舊記憶。
    $$f_t = \sigma(x_t W_{if}^T + b_{if} + h_{t-1} W_{hf}^T + b_{hf})$$
2.  **輸入門 ($i_t$)**：決定要更新哪些新資訊。
    $$i_t = \sigma(x_t W_{ii}^T + b_{ii} + h_{t-1} W_{hi}^T + b_{hi})$$
3.  **候選細胞狀態 ($\tilde{C}_t$)**：準備寫入細胞狀態的新候選值。
    $$\tilde{C}_t = \tanh(x_t W_{ig}^T + b_{ig} + h_{t-1} W_{hg}^T + b_{hg})$$
4.  **細胞狀態更新 ($C_t$)**：舊記憶乘以遺忘門，加上新記憶乘以輸入門。
    $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
5.  **輸出門 ($o_t$)**：決定下一個隱藏狀態 $h_t$ 應該輸出哪些部分。
    $$o_t = \sigma(x_t W_{io}^T + b_{io} + h_{t-1} W_{ho}^T + b_{ho})$$
6.  **隱藏狀態輸出 ($h_t$)**：將更新後的細胞狀態通過 $\tanh$，再與輸出門相乘。
    $$h_t = o_t \odot \tanh(C_t)$$

> [!NOTE]
> 由於 $f_t$ 的值域在 $[0, 1]$ 之間。當 $f_t \approx 1$ 時，舊的細胞狀態 $C_{t-1}$ 的梯度便能無損地流向 $C_t$，這徹底解決了長序列訓練中梯度消失的問題。

---

## 第二章：機器視覺的基石 (CNN) — 空間與特徵的提取

與序列資料不同，圖像資料具有強烈的**空間局部相關性**（例如：相鄰的像素通常代表同一個物體的一部分）。如果用全連接層來處理圖像，參數數量會隨圖像解析度呈二次方爆炸增長，且容易失去空間相對位置。這時就需要**卷積神經網路（Convolutional Neural Networks, CNN）**。

### 2.1 卷積（Convolution）與互相關（Cross-Correlation）

在深度學習中，我們通常說的「卷積層」在數學上實際上是**互相關運算**。它的核心特點是**局部連接（Local Connectivity）**與**權重共享（Parameter Sharing）**。

#### 2D 卷積計算過程
我們拿一個大小為 $(K_h, K_w)$ 的**卷積核（Kernel）**在圖像上滑動，每到一個位置，就將卷積核與圖像重疊區域的數值進行**按元素相乘並求和**，最後加上偏置（Bias）：

$$Y_{b, c_{out}, i, j} = b_{c_{out}} + \sum_{c_{in}=0}^{C_{in}-1} \sum_{m=0}^{K_h-1} \sum_{n=0}^{K_w-1} X_{b, c_{in}, i \cdot S + m, j \cdot S + n} \cdot W_{c_{out}, c_{in}, m, n}$$

```
Input Image (5x5)            Kernel (3x3)             Output (3x3)
┌───┬───┬───┬───┬───┐                                 ┌───┬───┬───┐
│ 1 │ 1 │ 1 │ 0 │ 0 │        ┌───┬───┬───┐            │ 4 │ 3 │ 4 │
├───┼───┼───┼───┼───┤        │ 1 │ 0 │ 1 │            ├───┼───┼───┤
│ 0 │ 1 │ 1 │ 1 │ 0 │   *    ├───┼───┼───┤     =      │ 2 │ 4 │ 3 │
├───┼───┼───┼───┼───┤        │ 0 │ 1 │ 0 │            ├───┼───┼───┤
│ 0 │ 0 │ 1 │ 1 │ 1 │        ├───┼───┼───┤            │ 2 │ 3 │ 4 │
├───┼───┼───┼───┼───┤        │ 2 │ 1 │ 0 │            └───┴───┴───┘
│ 0 │ 0 │ 1 │ 1 │ 0 │        └───┴───┴───┘
└───┴───┴───┴───┴───┘
```

#### 重要超參數
*   **Stride (步長, $S$)**：卷積核每次滑動的像素格子數。步長越大，輸出特徵圖的解析度越低。
*   **Padding (填充, $P$)**：在圖像邊緣填充 0 的圈數。主要是為了解決圖像邊緣資訊提取不足，以及控制輸出特徵圖的維度。

#### 輸出解析度計算公式
$$H_{out} = \left\lfloor \frac{H_{in} - K_h + 2P_h}{S_h} \right\rfloor + 1$$

---

## 第三章：注意力機制 (Attention) — 從 MHA 到大模型的 GQA 與 MQA

不論是 RNN 還是 CNN，它們提取資訊的感受野都是受限的。**注意力機制（Attention Mechanism）** 改變了這一切。它允許網路在一步之內，動態地將焦點對準序列中的任何一個位置，實現全域（Global）上下文關聯。

### 3.1 核心概念：Query, Key, Value
注意力機制可以類比為**資料庫查詢**：
*   **Query ($Q$)**：當前正在查詢的資訊。
*   **Key ($K$)**：各個位置所擁有的特徵標籤。
*   **Value ($V$)**：各個位置實際包含的內容資訊。

數學公式為：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$
其中 $\sqrt{d_k}$ 是縮放因子，用以防止點積值過大導致 Softmax 飽和而發生梯度消失；$M$ 是遮罩（Mask）。

---

### 3.2 現代大模型（LLM）的注意力進化

為了提升大語言模型在推理階段（Generation/Inference）的效率，注意力機制經歷了三次重要演進：

#### 1. Multi-Head Attention (MHA - 多頭注意力)
*   **做法**：Query、Key、Value 都各自擁有 $H$ 個獨立的注意力頭。
*   **缺點**：推理時需要快取每一個 Token 的 Key 和 Value (即 **KV Cache**)，這會消耗極大的記憶體帶寬。

#### 2. Multi-Query Attention (MQA - 多查詢注意力)
*   **做法**：所有的 Query 頭共享**同一個** Key 頭和 Value 頭。
*   **優缺點**：極大地降低了 KV Cache 的記憶體佔用，但因為多個頭只能共享一組 Key/Value 資訊，模型表達能力有所下降。

#### 3. Grouped-Query Attention (GQA - 分組查詢注意力)
*   **做法**：將 Query 頭分組（例如每 4 個或 8 個一組），每一組共享一組 Key 頭和 Value 頭。
*   **結論**：這是 LLaMA 3 與 Mistral 等現代開源大模型的標配，完美平衡了計算精度與記憶體儲存效率。

```
Multi-Head Attention (MHA)    Grouped-Query (GQA)         Multi-Query (MQA)
    Q1 Q2 Q3 Q4                   Q1 Q2 Q3 Q4                 Q1 Q2 Q3 Q4
    │  │  │  │                    └─┬─┘  └─┬─┘                └───┬───┘
    ▼  ▼  ▼  ▼                      ▼      ▼                      ▼
    K1 K2 K3 K4                     K1     K2                     K1
    V1 V2 V3 V4                     V1     V2                     V1
```

---

## 第四章：現代大語言模型 (LLM) 的穩定與加速技術 — RMSNorm, SwiGLU 與 RoPE

現代 LLM 的參數規模巨大（數十億到數千億），這使得原始 Transformer（如 2017 年的 Post-LN Transformer）在訓練時極易崩潰。以下三項關鍵技術是近年來大模型能夠成功訓練的基石。

### 4.1 RMSNorm (均方根歸一化)

傳統的 LayerNorm 通過計算特徵的均值和變異數來穩定激活值分佈：
$$\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$

論文研究發現，LayerNorm 的穩定性主要來自於**分母（變異數）的縮放**，而分子扣除均值（$\mu$）的平移操作對模型的穩定性貢獻不大，卻增加了計算開銷。
**RMSNorm** 去除了均值計算，直接使用 Root Mean Square 進行縮放：

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \odot \gamma$$
$$\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}$$

這使得大模型每一層的向前與向後傳播速度提升了 10%～50%。

---

### 4.2 SwiGLU (Swish 門控線性單元)

傳統 Transformer 使用兩層 Linear 搭配 ReLU 激活函數作為前饋網路（FFN）：
$$\text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2$$

現代大模型使用 **SwiGLU** 替代它。GLU（Gated Linear Unit）的核心是**門控機制**：一個線性分支負責處理資訊，另一個線性分支通過非線性激活後作為「開關」，控制資訊的流動。
SwiGLU 選擇使用 Swish（在 PyTorch 中常稱為 SiLU）作為其激活函數：

$$\text{SwiGLU}(x) = (\text{Swish}(x W_{gate}) \odot x W_{up}) W_{down}$$

> [!TIP]
> SwiGLU 通過增加一個權重矩陣（原本是兩個，現在是三個），提供了更平滑的非線性表達能力與更穩定的梯度，已被證明能顯著加快模型的收斂速度。

---

### 4.3 RoPE (旋轉位置編碼)

傳統的正弦位置編碼是**加算式**（Additive）的，這使得注意力機制在計算 Query 和 Key 的內積時，相對位置資訊很容易在矩陣相乘中被「稀釋」或打亂。

**RoPE（Rotary Position Embedding）** 提出了一種極其優雅的方法：將 2D 平面中的向量旋轉一個角度。
具體來說，它將 Query 和 Key 向量每兩個維度分組，視為複數平面上的一個點，並根據其所在的序列絕對位置 $m$ 進行旋轉：

$$\text{RoPE}(x_m) = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} x_{m, 1} \\ x_{m, 2} \end{pmatrix}$$

#### 為什麼 RoPE 能自然捕捉相對距離？
由於旋轉矩陣具有正交性，旋轉後的 Query $q_m$ 與 Key $k_n$ 的內積為：
$$(\mathbf{R}_{m} q)^T (\mathbf{R}_{n} k) = q^T \mathbf{R}_{m}^T \mathbf{R}_{n} k = q^T \mathbf{R}_{n - m} k$$
內積結果**只與它們的相對距離 $n - m$ 有關**！這賦予了模型極強的外推能力（Extrapolation），也就是說，模型即使在短序列上訓練，也能自然推展到較長的文本長度。

---

## 第五章：經典 Transformer 與 Decoder-Only LLM 的組裝

在掌握了上述組件後，我們可以將它們組裝成兩種架構：

1.  **經典 Seq2Seq Transformer**
    *   **結構**：包含獨立的 Encoder（解讀源序列）與 Decoder（藉由 Cross-Attention 參照 Encoder 的輸出，自迴歸生成目標序列）。
    *   **用途**：機器翻譯、文章摘要。
2.  **Decoder-Only LLM (現代大模型主流)**
    *   **結構**：只有解碼器堆疊。所有 Token 都在一個統一的序列中，使用 Causal Mask 防止未來資訊洩漏。
    *   **技術棧**：Pre-LN (RMSNorm) + GQA + RoPE + SwiGLU。
    *   **用途**：ChatGPT, LLaMA, Claude 等通用生成式對話模型。

---

## 附錄：如何使用本專案進行自我驗證 (測試驅動學習)

本專案目錄內包含一整套對齊測試代碼。這就是你的 **Superpowers**：

### 驗證步驟

1.  **安裝測試環境**：
    確保處於專案根目錄下，安裝所需的套件：
    ```bash
    pip install -r requirements.txt
    ```
2.  **執行測試驗證**：
    當你開始在 `src/transformer/` 中撰寫程式碼時，可以隨時使用以下指令檢驗進度：
    ```bash
    pytest tests/test_transformer.py
    ```
3.  **觀看測試結果**：
    *   **Fail / Error**：表示你的數學實作、矩陣乘法維度或偏置相加有誤。
    *   **Pass (綠燈)**：恭喜！代表你的自製 Transformer/LLM 組件與 LLaMA 等前沿架構的理論對齊一致。

---

祝你學習愉快！現在，就從編輯 [rnn_cell.py](file:///Users/yuhan/coding/ai_knowledge/src/rnn/rnn_cell.py) 或 [attention.py](file:///Users/yuhan/coding/ai_knowledge/src/transformer/attention.py) 開始你的實作之旅吧！
