# 從零開始：自手實作深度學習與大語言模型核心架構 (終極學習與實作指南)

歡迎閱讀這本書！本書專為希望藉由「從零開始手寫實作（From Scratch）」來真正吃透現代深度學習與大語言模型（LLM）底層機制的研究者與工程師所寫。

我們拒絕僅停留在理論公式，也拒絕直接呼叫高階套件。我們將結合 **OpenSpec（規格驅動）** 與 **Superpowers（測試驅動）** 的軟體工程邏輯，把數學原理一步步轉化為能跑、可過單元測試的 PyTorch 程式碼。

---

## 📖 本書大綱

*   [第一章：自動求導 (Autograd) 與計算圖 — 深度學習的物理心臟](#第一章自動求導-autograd-與計算圖--深度學習的物理心臟)
*   [第二章：基礎層與激活函數 (GELU) — 讓網路具備非線性表達能力](#第二章基礎層與激活函數-gelu--讓網路具備非線性表達能力)
*   [第三章：優化器的藝術 (AdamW) — 訓練大語言模型的黃金鑰匙](#第三章優化器的藝術-adamw--訓練大語言模型的黃金鑰匙)
*   [第四章：循環神經網路 (RNN & LSTM) — 時間與序列的建模](#第四章循環神經網路-rnn--lstm--時間與序列的建模)
*   [第五章：卷積神經網路 (CNN) — 空間特徵提取與下採樣](#第五章卷積神經網路-cnn--空間特徵提取與下採樣)
*   [第六章：注意力機制 (Attention) — 從 MHA 到大模型的 GQA 與 MQA](#第六章注意力機制-attention--從-mha-到大模型的-gqa-與-mqa)
*   [第七章：現代大語言模型核心組件 — RMSNorm, SwiGLU 與 RoPE](#第七章現代大語言模型核心組件--rmsnorm-swiglu-與-rope)
*   [第八章：LLM 推理效能優化 (KV Cache) — 推理加速的神奇魔法](#第八章llm-推理效能優化-kv-cache--推理加速的神奇魔法)
*   [附錄：測試驅動開發 (TDD) 與驗證指南](#附錄測試驅動開發-tdd-與驗證指南)

---

## 第一章：自動求導 (Autograd) 與計算圖 — 深度學習的物理心臟

深度學習的本質是透過梯度下降法（Gradient Descent）來尋找模型最優權重。而計算梯度（Gradient）的底層機制，就是「自動求導（Autograd）」。

### 1.1 計算圖（Computational Graph）的數學本質
計算圖是將數學表達式轉化為有向無環圖（DAG）的表示方法：
*   **節點 (Nodes)**：代表變數（如輸入 $x$、權重 $w$、偏置 $b$）或運算運算元（如 $+$, $\times$, $\exp$, $\log$）。
*   **邊 (Edges)**：代表運算的依賴關係與資料流向。

#### 前向傳播 (Forward Pass)
前向傳播是沿著計算圖的有向邊，從輸入節點開始，依序計算各中間節點的值，直到輸出損失值（Loss）。
例如，計算 $z = (x + y) \times w$ 的過程：
1.  定義中間變數：$u = x + y$
2.  計算最終輸出：$z = u \times w$

#### 反向傳播 (Backward Pass) 與鏈式法則 (Chain Rule)
反向傳播是從輸出節點（Loss）開始，逆向計算 Loss 對於圖中每個節點的偏微分。
根據微積分的鏈式法則：
$$\frac{\partial z}{\partial x} = \frac{\partial z}{\partial u} \cdot \frac{\partial u}{\partial x}$$

由於 $u = x + y$，其局部偏微分（Local Gradient）為：
$$\frac{\partial u}{\partial x} = 1, \quad \frac{\partial u}{\partial y} = 1$$

由於 $z = u \times w$，其局部偏微分為：
$$\frac{\partial z}{\partial u} = w, \quad \frac{\partial z}{\partial w} = u$$

因此， Loss 對於輸入 $x$ 的梯度為：
$$\frac{\partial z}{\partial x} = w \cdot 1 = w$$

```
前向計算 (左到右):
x (2) ──┐
        ├──► u = x + y (5) ──┐
y (3) ──┘                    ├──► z = u * w (20)
                      w (4) ─┘

反向傳播 (右到左):
dz/dx = 4 ◄──┐
             ├── dz/du = 4 ◄─── dz/dz = 1
dz/dy = 4 ◄──┘
                      dz/dw = 5 ◄──┘
```

> [!NOTE]
> 在動態計算圖（如 PyTorch 的 Autograd）中，當前向傳播執行時，系統會自動在記憶體中建立一個計算圖。每個張量（Tensor）都會保存指向建立它的算子節點的指標（`grad_fn`）。當呼叫 `loss.backward()` 時，引擎就會沿著這些指標逆向遍歷計算圖，自動套用鏈式法則算出所有梯度。

---

### 1.3 互動式自動求導與計算圖視覺化工具
您可以調整下方滑動條改變輸入與權重，並點選「前向傳播」與「反向傳播」觀察數據與梯度的傳遞路徑與粒子流向動畫。

<div id="autograd-visualizer" class="interactive-visualizer"></div>

---

## 第二章：基礎層與激活函數 (GELU) — 讓網路具備非線性表達能力

### 2.1 線性投影層（Linear Layer）的維度與梯度傳遞
線性層（亦稱全連接層）執行的是仿射變換：
$$Y = X W^T + b$$

*   $X$：輸入矩陣，維度為 $(B, D_{in})$，其中 $B$ 是 Batch Size，$D_{in}$ 是輸入特徵維度。
*   $W$：權重矩陣，維度為 $(D_{out}, D_{in})$。
*   $b$：偏置向量，維度為 $(D_{out})$。
*   $Y$：輸出矩陣，維度為 $(B, D_{out})$。

#### 為什麼要對 $W$ 進行轉置？
因為在數學上，我們習慣將輸入向量表示為列向量（Column Vector），即 $y = W x + b$。但在寫程式實作時，為了配合矩陣批次運算，資料通常以列矩陣（Row Matrix）排列，故改寫為 $Y = X W^T + b$。

#### 手寫梯度求導（Backpropagation through Linear Layer）
假設從上游傳回的 Loss 對於 $Y$ 的梯度為 $dY \in \mathbb{R}^{B \times D_{out}}$：
1.  **對輸入 $X$ 的梯度**：
    $$dX = dY \cdot W \quad (\text{維度: } (B, D_{in}) = (B, D_{out}) \times (D_{out}, D_{in}))$$
2.  **對權重 $W$ 的梯度**：
    $$dW = dY^T \cdot X \quad (\text{維度: } (D_{out}, D_{in}) = (D_{out}, B) \times (B, D_{in}))$$
3.  **對偏置 $b$ 的梯度**（在 Batch 維度上求和）：
    $$db = \sum_{i=1}^B dY_i \quad (\text{維度: } (D_{out}))$$

---

### 2.2 非線性激活函數的必要性
如果沒有非線性激活函數，多個線性層疊加的結果仍然只是一個線性變換：
$$Y = ((X W_1 + b_1) W_2 + b_2) = X (W_1 W_2) + (b_1 W_2 + b_2) = X W_{eff} + b_{eff}$$
這使得網路無法學習高維度的複雜非線性曲面（如圖像分類邊界、文本邏輯）。

#### 1. ReLU (Rectified Linear Unit)
$$\text{ReLU}(x) = \max(0, x)$$
*   **優點**：計算簡單，且在大於 0 的區間內梯度恆為 1，極大程度避免了深層網路中的「梯度消失」問題。
*   **缺點 (Dying ReLU)**：當輸入小於 0 時，梯度為 0。如果一個神經元的輸入在多個 Batch 中都為負數，它的權重將永遠不會更新，該神經元會永久「壞死」。

#### 2. GELU (Gaussian Error Linear Unit)
現代前沿 Transformer（如 GPT-3/4、LLaMA、BERT）的標準配備。
*   **數學思想**：GELU 將隨機正則化（如 Dropout）的思想融入激活函數中。它並不像 ReLU 那樣生硬地在 0 點切斷，而是根據輸入值 $x$ 的大小，依據高斯分佈（正態分佈）的累積分布函數（CDF）$\Phi(x)$，來決定該特徵被保留的機率：
    $$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot P(X \le x), \quad X \sim \mathcal{N}(0, 1)$$
*   由於正態分佈的 CDF 無法用解析幾何公式表達，在硬體實作時常使用雙曲正切（$\tanh$）來做極為精準的近似：
    $$\text{GELU}(x) \approx 0.5x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \left(x + 0.044715 x^3\right)\right)\right)$$

```
GELU vs ReLU 曲線對比:
  y ^
    │               / GELU (平滑過渡)
    │              /
    │   _ ─── ── ─/──► ReLU (0處折線)
────┼────────────/────► x
    │
```

---

## 第三章：優化器的藝術 (AdamW) — 訓練大語言模型的黃金鑰匙

參數的更新方式決定了模型能否收斂以及收斂的速度。

### 3.1 從 SGD 到 Adam
Stochastic Gradient Descent (SGD) 沿著當前 Batch 的梯度反方向更新：
$$\theta_{t+1} = \theta_t - \eta \cdot g_t$$
這在非凸函數的「峽谷」地形中，會導致在兩側陡峭牆壁來回震盪，而在平緩谷底前進極慢。

**Adam（Adaptive Moment Estimation）** 結合了兩大技術來克服這個問題：
1.  **一階動量 (Momentum)**：記錄過去梯度的指數移動平均（EMA），像是有慣性的球，能衝過局部極小值（Local Minima）。
2.  **二階動量 (自適應學習率)**：記錄梯度平方的移動平均，藉此估計每個參數的稀疏度。對於梯度較小的參數給予較大的更新步幅，對於頻繁更新的參數則予以抑制。

---

### 3.2 為什麼現代大模型必須用 AdamW？
在大語言模型訓練中，我們通常會使用 L2 正則化（Weight Decay）來防止權重無限膨脹（過擬合）。
在傳統的 Adam 優化器中，L2 正則化是直接加在梯度 $g_t$ 上：
$$g_t = g_t + \lambda \theta_t$$
這會使正則化項被混入一階矩 $m_t$ 與二階矩 $v_t$ 的計算中。因為二階矩的分母縮放，導致權重衰減（Weight Decay）的程度與梯度大小掛鉤：梯度大的參數衰減慢，梯度小的參數反而衰減快，這在數學上不符合正則化本意。

**AdamW** 修正了這一問題，將**權重衰減從梯度的更新步驟中解耦（Decoupled）**，直接作用在參數更新的最後一步：

#### AdamW 演算法步驟
在每個時間步 $t$：
1.  **計算梯度**：$g_t = \nabla_{\theta} L(\theta_t)$
2.  **更新一階動量（動量）**：
    $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
3.  **更新二階動量（自適應大小）**：
    $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
4.  **偏差修正（Bias Correction）**（因為 $m_0, v_0$ 初始化為 0，在初期會偏向 0，需要修正）：
    $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
5.  **解耦權重衰減並更新參數**：
    $$\theta_{t+1} = \theta_t - \eta \cdot \lambda \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

其中：
*   $\eta$：學習率。
*   $\lambda$：權重衰減率。
*   $\beta_1, \beta_2$：動量衰減因子（通常設為 0.9 與 0.999）。
*   $\epsilon$：數值穩定常數（通常為 $1\times 10^{-8}$）。

---

## 第四章：循環神經網路 (RNN & LSTM) — 時間與序列的建模

### 4.1 傳統 RNN 的設計與限制
傳統 RNN 藉由公式傳遞狀態：
$$h_t = \tanh(X_t W_{ih}^T + b_{ih} + h_{t-1} W_{hh}^T + b_{hh})$$

在反向傳播時，對於極長序列，隱藏狀態 $h_t$ 對很久以前的 $h_0$ 的偏微分會包含 $W_{hh}^T$ 的連乘積。這會導致：
*   若 $W_{hh}$ 最大特徵值 $> 1$：梯度呈指數級增長，發生 **梯度爆炸**。
*   若 $W_{hh}$ 最大特徵值 $< 1$：梯度呈指數級衰減，發生 **梯度消失**。

---

### 4.2 LSTM 的「傳送帶」與三大門控機制
為了克服梯度消失，LSTM 引入了 **Cell State ($C_t$)**。
梯度可以在這條通道上無阻礙流動。而門控（Gates）則使用 Sigmoid 函數輸出 $[0, 1]$ 區間的值，來決定資訊的流出流入量：

```
                    細胞狀態 Cell State (C_t)
   C_{t-1} ───────────────────⊕─────────────────── C_t
                             ▲
                             │ (i_t * g_t)
              ┌───┐        ┌─┴─┐        ┌───┐
   h_{t-1} ──►│ f ├──────► │ i ├──────► │ o ├───► 隱藏狀態 h_t
              └───┘        └───┘        └───┘
              遺忘門        輸入門        輸出門
```

#### 六大核心更新公式解密
1.  **遺忘門 ($f_t$)**：決定上一時刻的細胞狀態 $C_{t-1}$ 有多少比例要保留下來。
    $$f_t = \sigma(X_t W_{if}^T + b_{if} + h_{t-1} W_{hf}^T + b_{hf})$$
2.  **輸入門 ($i_t$)**：決定當前輸入的資訊有多少要寫入細胞狀態。
    $$i_t = \sigma(X_t W_{ii}^T + b_{ii} + h_{t-1} W_{hi}^T + b_{hi})$$
3.  **候選細胞狀態 ($\tilde{C}_t$)**：當前步所產生的全新記憶內容。
    $$\tilde{C}_t = \tanh(X_t W_{ig}^T + b_{ig} + h_{t-1} W_{hg}^T + b_{hg})$$
4.  **細胞狀態更新 ($C_t$)**：新舊記憶交融。
    $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
5.  **輸出門 ($o_t$)**：決定當前的細胞狀態有多少要轉化為隱藏狀態 $h_t$ 輸出。
    $$o_t = \sigma(X_t W_{io}^T + b_{io} + h_{t-1} W_{ho}^T + b_{ho})$$
6.  **隱藏狀態 ($h_t$)**：
    $$h_t = o_t \odot \tanh(C_t)$$

---

## 第五章：卷積神經網路 (CNN) — 空間特徵提取與下採樣

對於影像等二維資料，全連接層會忽略像素之間的局部空間位置資訊。卷積網路透過**局部連接**與**權重共享**克服了這個問題。

### 5.1 2D 卷積計算細節
卷積運算本質上是滑動視窗內的「點積並相加」：
$$Y_{b, c_{out}, i, j} = b_{c_{out}} + \sum_{c_{in}=0}^{C_{in}-1} \sum_{m=0}^{K_h-1} \sum_{n=0}^{K_w-1} X_{b, c_{in}, i \cdot S + m, j \cdot S + n} \cdot W_{c_{out}, c_{in}, m, n}$$

```
滑動視窗計算過程 (Stride=1, Padding=0):
Input (4x4)                Kernel (2x2)             Output (3x3)
┌───┬───┬───┬───┐
│ 1 │ 2 │ 0 │ 1 │          ┌───┬───┐                ┌───┬───┬───┐
├───┼───┼───┼───┤          │ 1 │ 0 │                │ 3 │ 4 │ 1 │
│ 0 │ 1 │ 2 │ 0 │     *    ├───┼───┤         =      ├───┼───┼───┤
├───┼───┼───┼───┤          │ 2 │ 1 │                │ 2 │ 6 │ 4 │
│ 1 │ 0 │ 2 │ 1 │          └───┴───┘                ├───┼───┼───┤
├───┼───┼───┼───┤                                   │ 3 │ 4 │ 5 │
│ 0 │ 1 │ 1 │ 1 │                                   └───┴───┴───┘
└───┴───┴───┴───┘
(計算第一個元素: 1*1 + 2*0 + 0*2 + 1*1 = 3)
```

*   **Padding (填充)**：在輸入邊緣補 0，避免邊角特徵在多次卷積後快速流失，並能控制輸出大小。
*   **Stride (步長)**：卷積核每次移動的步長。步長增加會以倍數縮減輸出特徵圖的尺寸。

---

### 5.2 池化層 (Pooling) 的數學效應
最大池化（Max Pooling）在一個固定視窗內取最大值：
$$Y_{b, c, i, j} = \max_{m, n} X_{b, c, i \cdot S + m, j \cdot S + n}$$
*   **平移不變性 (Translation Invariance)**：如果特徵在影像中輕微移動，只要它仍在同一個池化窗口內，池化層的輸出就不會改變。
*   **降維與防過擬合**：顯著縮減空間維度，減少後續層的運算參數。

---

### 5.3 互動式 2D 卷積滑動核視覺化工具
點擊下方的播放按鈕，觀察卷積核如何在 5x5 的圖像上滑動，計算出 3x3 的特徵圖結果。

<div id="cnn-visualizer" class="interactive-visualizer"></div>

---

## 第六章：注意力機制 (Attention) — 從 MHA 到大模型的 GQA 與 MQA

注意力機制擺脫了卷積和循環在長距離關聯上的物理限制，讓模型能「一步到位」地存取全局資訊。

### 6.1 自注意力（Self-Attention）的公式推導
給定長度為 $L$、嵌入維度為 $d_{model}$ 的輸入向量矩陣 $X \in \mathbb{R}^{B \times L \times d_{model}}$：

1.  **投影生成 $Q, K, V$**：
    $$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$
    where $W_Q, W_K, W_V \in \mathbb{R}^{d_{model} \times d_{model}}$。
2.  **計算相似度（點積注意力）**：
    $$A = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right)$$
    *   $QK^T$ 的每個元素代表第 $i$ 個 Token 和第 $j$ 個 Token 之間的相關度。
    *   除以 $\sqrt{d_k}$ 是為了防止點積數值過大，導致 Softmax 進入飽和區（梯度近乎為 0）。
    *   $M$ 是 Causal Mask，形狀為 $(L, L)$，上三角填入 $-\infty$，下三角為 0。
3.  **加權 Value 矩陣**：
    $$\text{Output} = A V$$

---

### 6.2 現代 LLM 注意力機制的演進
在 LLM 生成 Token 時，我們需要快取過去所有 Token 的 Key 和 Value，稱為 **KV Cache**。
對於大模型，KV Cache 的記憶體佔用極大，甚至會超過模型本身的參數大小。為此，注意力結構進行了升級：

1.  **Multi-Head Attention (MHA)**：
    每個 Query 頭對應自己獨立的 Key 和 Value 頭。
    *   *KV 快取佔用*：大（每一層、每個頭、每個 Token 都要存一組 Key/Value 向量）。
2.  **Multi-Query Attention (MQA)**：
    所有的 Query 頭共享同一個 Key 頭和 Value 頭。
    *   *KV 快取佔用*：小（降低至原本的 $1 / H_{heads}$）。
    *   *缺點*：模型表達能力顯著退化，難以收斂。
3.  **Grouped-Query Attention (GQA)**：
    Query 頭被分為數個組，每組共享一個 Key 頭 and Value 頭。
    *   *優點*：完美融合了 MHA 的高品質與 MQA 的超低記憶體佔用。**LLaMA 3、Mistral 等現代 LLM 皆全面採用 GQA。**

```
MHA (多頭)                     GQA (分組)                  MQA (單一)
Q1 Q2 Q3 Q4                    Q1 Q2 Q3 Q4                 Q1 Q2 Q3 Q4
│  │  │  │                     └─┬─┘  └─┬─┘                 └───┬───┘
▼  ▼  ▼  ▼                       ▼      ▼                       ▼
K1 K2 K3 K4                      K1     K2                      K1
V1 V2 V3 V4                      V1     V2                      V1
```

---

### 6.3 互動式自注意力矩陣權重視覺化工具
將滑鼠懸停在不同的單字（Query）上，觀察自注意力機制如何為各個 Token 分配不同的權重強度（以顏色深淺表示）。

<div id="attention-visualizer" class="interactive-visualizer"></div>

---

## 第七章：現代大語言模型核心組件 — RMSNorm, SwiGLU 與 RoPE

為了支撐百億至千億級參數量的穩定訓練與極長文本推理，現代 LLM 引進了以下核心技術：

### 7.1 RMSNorm (均方根歸一化)
在 Pre-LN 架構中，我們需要在注意力與前饋層前對輸入進行歸一化。
傳統 LayerNorm 的公式為：
$$\text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$
**RMSNorm** 發現，去掉減去均值 $\mu$ 和偏置 $\beta$ 的步驟，不僅不會降低收斂穩定性，還能減少大量的均值與減法計算，提升約 10%~20% 的運算速度：
$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}$$

---

### 7.2 SwiGLU FFN (門控前饋網路)
傳統前饋網路採用兩層線性層配 ReLU。
**SwiGLU** 採用了**門控線性單元 (GLU)** 架構，並以 Swish (SiLU) 作為激活函數：
$$\text{SwiGLU}(x) = (\text{Swish}(x W_{gate}) \odot x W_{up}) W_{down}$$
*   $W_{gate}$ 分支相當於一個「開關」，控制資訊通過 $W_{up}$ 分支的比例。這使得梯度流動更加平滑，經驗表現顯著優於 ReLU。

---

### 7.3 RoPE (旋轉位置編碼)
傳統 Transformer 將位置編碼直接加在輸入 Embedding 上。這對點積注意力中計算相對位置不夠直觀。
**RoPE** 直接在 Query 和 Key 投影後，對向量的每兩個維度進行二維平面旋轉：

$$\text{RoPE}(x_m) = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} x_{m, 1} \\ x_{m, 2} \end{pmatrix}$$

#### 相對距離不變性
由於旋轉矩陣具有正交性，旋轉後的 Query $q_m$ 與 Key $k_n$ 的點積為：
$$(\mathbf{R}_{m} q)^T (\mathbf{R}_{n} k) = q^T \mathbf{R}_{m}^T \mathbf{R}_{n} k = q^T \mathbf{R}_{n - m} k$$
這代表**兩個 Token 之間的注意力權重，只與它們的相對距離 $n - m$ 有關**。這使得 RoPE 具備極強的長文本外推能力（Extrapolation）。

---

## 第八章：LLM 推理效能優化 (KV Cache) — 推理加速的神奇魔法

### 8.1 為什麼自迴歸解碼 (Auto-regressive) 越來越慢？
大語言模型是逐個 Token 生成的。當我們要生成第 $t$ 個 Token 時，輸入是前面已生成的整個序列 $X_{1:t-1}$。
在計算自注意力時，我們需要計算：
$$Q = X_{1:t-1} W_Q, \quad K = X_{1:t-1} W_K, \quad V = X_{1:t-1} W_V$$
這代表每一次生成新 Token，我們都要把舊的 Prompt 重頭計算一遍。這會導致：
*   **計算量呈 $O(L^2)$ 增長**，生成越到後面越卡頓。

---

### 8.2 KV Cache 的工作原理
我們注意到，在前向傳播中，過去已經生成的 Token，其對應的 Key 和 Value 向量是**固定不變的**。
因此，我們可以：
1.  **快取 (Cache)** 過去所有步驟計算出的 $K$ 和 $V$。
2.  **單步計算**：在當前步驟 $t$，我們**只將當前的新 Token** $x_t$ 輸入網路，投影出當前的單個 $q_t, k_t, v_t$。
3.  **動態拼接**：將新算出的 $k_t, v_t$ 拼接到快取矩陣中：
    $$K_{cached} \leftarrow [K_{cached}, k_t], \quad V_{cached} \leftarrow [V_{cached}, v_t]$$
4.  **計算注意力**：僅用當前的 $q_t$ 與整條 $K_{cached}$ 計算相關權重，並與 $V_{cached}$ 相乘。

這將單個 Token 的生成複雜度從 $O(L^2)$ 降低到 **$O(L)$**，是所有現代大模型推理引擎（如 vLLM、TensorRT-LLM）必備的核心加速技術。

---

## 附錄：測試驅動開發 (TDD) 與驗證指南

現在專案結構已重構完畢，您可以使用我們安排的單元測試，來進行自我實作檢驗。

1.  **安裝依賴**：
    ```bash
    pip install -r requirements.txt
    ```
2.  **執行基礎單元測試**：
    當您寫完基礎層或優化器後，請執行：
    ```bash
    pytest tests/test_foundations.py
    ```
3.  **執行 Transformer 測試**：
    ```bash
    pytest tests/test_transformer.py
    ```

綠燈（Pass）代表您的自製組件在數學上完全正確。祝您手寫 AI 順利！
