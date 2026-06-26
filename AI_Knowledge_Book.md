# 從零開始：自手實作深度學習與大語言模型核心架構 (手把手學習與實作指南)

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

要理解深度學習，首先要理解「機器是如何學習的」。其本質是：**前向傳播計算誤差 $\rightarrow$ 計算誤差對參數的偏微分（梯度） $\rightarrow$ 沿著梯度反方向微調參數。**

### 1.1 什麼是計算圖（Computational Graph）？
計算圖是一種將複雜的數學運算表示為「節點（Node）」與「有向邊（Edge）」的圖形。
*   **節點**：代表運算（如加、乘、$\sin$、$\exp$）或變數（參數、輸入）。
*   **有向邊**：代表資料的流動方向。

例如，計算 $z = (x + y) \times w$ 的計算圖如下：
```
  x ──┐
      ├──► [ + ] (u) ──┐
  y ──┘                ├──► [ × ] (z)
                  w ───┘
```
在**前向傳播**時，我們從左向右計算數值。在**反向傳播**時，我們從右向左，利用**微積分的鏈式法則（Chain Rule）**，動態地將上游梯度乘以當前節點的局部偏微分，流向下游。

### 1.2 鏈式法則（Chain Rule）
假設 $z$ 是 $u$ 的函數，而 $u$ 是 $x$ 的函數，則 $z$ 對 $x$ 的偏微分為：
$$\frac{\partial z}{\partial x} = \frac{\partial z}{\partial u} \times \frac{\partial u}{\partial x}$$

在自動求導（Autograd）引擎中，每個節點只需專注於計算自己的**局部導數（Local Gradient）**：
1.  節點收到來自上游（右邊）傳回的梯度 $\frac{\partial \text{Loss}}{\partial \text{Output}}$。
2.  節點計算當前運算對輸入的局部偏微分 $\frac{\partial \text{Output}}{\partial \text{Input}}$。
3.  兩者相乘，將梯度傳遞給下游（左邊）：
    $$\frac{\partial \text{Loss}}{\partial \text{Input}} = \frac{\partial \text{Loss}}{\partial \text{Output}} \times \frac{\partial \text{Output}}{\partial \text{Input}}$$

> [!NOTE]
> 自動求導的精妙之處在於，不論網路多複雜，我們只需要定義好每個原子級別運算（加、減、乘、矩陣乘法等）的 forward 和 backward 規則，電腦就能以萬用的邏輯精準算出任何變數的梯度！

---

## 第二章：基礎層與激活函數 (GELU) — 讓網路具備非線性表達能力

### 2.1 全連接層 (Linear Layer)
全連接層是深度學習中最普遍的投影層：
$$Y = X W^T + b$$
*   **前向傳播**：進行矩陣乘法，將低維特徵投影至高維，或將高維特徵壓縮。
*   **反向傳播梯度傳遞**：
    *   $\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} W$
    *   $\frac{\partial L}{\partial W} = \left(\frac{\partial L}{\partial Y}\right)^T X$
    *   $\frac{\partial L}{\partial b} = \sum_{\text{batch}} \frac{\partial L}{\partial Y}$

---

### 2.2 為什麼需要激活函數？
如果沒有激活函數，不論我們疊加多少層線性層，多個矩陣相乘的結果依然只是一個單一的線性變換（$Y = X W_1 W_2 = X W_{net}$），這使得模型完全無法擬合複雜的非線性關係（例如 XOR 問題或影像分類）。

#### 1. ReLU (Rectified Linear Unit)
$$\text{ReLU}(x) = \max(0, x)$$
*   **特點**：計算極快，在正區間梯度為 1，有效緩解梯度消失；但若輸入一直小於 0，該神經元會永久壞死（Dying ReLU）。

#### 2. GELU (Gaussian Error Linear Unit)
**現代 LLM（GPT-4、BERT）的標配激活函數**。
*   **物理直覺**：與 ReLU 直截了當地在 0 點切斷不同，GELU 根據輸入值的大小，以概率決定是否保留該特徵。它結合了機率密度函數（高斯分佈），使非線性邊界更平滑。
*   由於正態分佈的累積分布函數計算複雜，硬體上常用以下公式近似實作：
    $$\text{GELU}(x) \approx 0.5x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \left(x + 0.044715 x^3\right)\right)\right)$$

---

## 第三章：優化器的藝術 (AdamW) — 訓練大語言模型的黃金鑰匙

計算出梯度後，我們需要更新參數。**優化器**決定了參數更新的步幅與方向。

### 3.1 隨機梯度下降 (SGD)
$$\theta_{t+1} = \theta_t - \eta \cdot g_t$$
*   **問題**：若所有參數都使用相同的學習率 $\eta$，在陡峭的方向容易震盪，在平緩的方向前進極慢。

### 3.2 AdamW 的三大支柱 (Decoupled Weight Decay)
為了加速收斂，Adam 結合了 **一階動量（Momentum，記憶過去的方向）** 與 **二階動量（自適應學習率，為頻繁更新的參數降低步長，為稀疏更新的參數增大步長）**。

而在大模型時代，我們通常需要使用 **L2 正則化（Weight Decay）** 防止過擬合。傳統的 Adam 會將 L2 正則化的梯度直接放入二階矩中計算，這會使得常更新的權重衰減變慢，不常更新的權重衰減變快。
**AdamW（Adam with Decoupled Weight Decay）** 修正了這一點，將權重衰減直接作用在參數更新的最後一步，與梯度的矩估計完全解耦：

```
1. 計算梯度 g_t ──────► 2. 更新一階與二階矩 (m_t, v_t) ──► 3. 計算偏差修正
                                                                    │
   ┌────────────────────────────────────────────────────────────────┘
   ▼
4. 更新參數: θ_{t+1} = θ_t - η * (λ * θ_t) - [ η / (sqrt(v_t) + ε) ] * m_t
                                  └── Decoupled Weight Decay ──┘
```

> [!IMPORTANT]
> 幾乎所有現代大型 Transformer 網路（LLaMA, GPT 系列）的訓練，都是使用 **AdamW** 作為優化器。它是大模型得以穩定收斂的底層支柱。

---

## 第四章：循環神經網路 (RNN & LSTM) — 時間與序列的建模

在自然語言等時間序列中，前後資料有強烈關聯，需要「短期記憶」。

### 4.1 傳統 RNN
$$h_t = \tanh(x_t W_{ih}^T + b_{ih} + h_{t-1} W_{hh}^T + b_{hh})$$
*   由於反向傳播在時間軸上連乘（BPTT），容易導致梯度呈指數級衰減（**梯度消失**）或爆炸，使模型無法記住 10 個時間步之前的資訊。

### 4.2 LSTM (長短期記憶)
引入**細胞狀態 ($C_t$)** 作為高速公路，並由三個門（Gates）來調控：
1.  **遺忘門 ($f_t$)**：決定丟棄多少舊記憶。
2.  **輸入門 ($i_t$)**：決定寫入多少新資訊。
3.  **輸出門 ($o_t$)**：決定隱藏狀態 $h_t$ 輸出什麼。

這使得梯度可以在細胞狀態通道上暢通無阻，大幅延長了記憶距離。

---

## 第五章：卷積神經網路 (CNN) — 空間特徵提取與下採樣

對於影像，像素的「鄰近關係」極度重要。

### 5.1 局部連接與權重共享
*   **局部連接**：卷積核每次只處理一小塊局部區域（感受野），保留空間結構。
*   **權重共享**：一個卷積核在整張影像上滑動，使用同一組參數。這不僅大幅減少參數，更讓模型具備「平移不變性」（無論貓在影像的左上角還是右下角，都能被同一個卷積核偵測到）。

---

## 第六章：注意力機制 (Attention) — 從 MHA 到大模型的 GQA 與 MQA

Transformer 完全捨棄了卷積與循環，僅依賴「注意力」來抓取上下文。

### 6.1 核心機制
將輸入向量投影為 Queries ($Q$), Keys ($K$), Values ($V$)。計算 Query 與所有 Key 的相似度（點積），經過 Softmax 歸一化後加權 Value：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$

*   **Causal Mask**：在解碼生成時，將未來位置的注意力權重設為 $-\infty$（使 Softmax 計算後為 0），防止模型偷看未來的答案。

### 6.2 大模型的推理優化：GQA 與 MQA
當大模型生成文本時，每個 Token 都要跟前面所有 Token 的 Key 和 Value 計算注意力。為了加速，我們會在記憶體中快取前面的 Key/Value，即 **KV Cache**。
*   **MHA**：每個 Query 頭有自己獨立的 KV 頭，KV Cache 佔用極大記憶體帶寬。
*   **MQA**：所有 Query 頭共享一個 KV 頭，節省帶寬但精度降低。
*   **GQA**：Query 分組共享 KV 頭，兼顧效率與準確度，是當前 LLaMA 3 等大模型的標配。

---

## 第七章：現代大語言模型核心組件 — RMSNorm, SwiGLU 與 RoPE

為了將 Transformer 擴展到千億參數，科學家對其進行了數次架構上的升級：

### 7.1 RMSNorm (均方根歸一化)
*   去除 LayerNorm 的均值減法，僅利用均方根進行尺度縮放，計算效率更高且訓練同樣穩定。

### 7.2 SwiGLU FFN
*   利用 Swish 激活函數的門控線性單元（SwiGLU），替代傳統前饋網路中的 ReLU 激活，帶來更平滑的梯度流與更強的擬合能力。

### 7.3 RoPE 旋轉位置編碼
*   直接在複數空間上對 Query 與 Key 進行向量旋轉，使點積運算直接包含「相對位置」資訊。這給予了現代 LLM 極為強大的**上下文長度外推能力**。

---

## 第八章：LLM 推理效能優化 (KV Cache) — 推理加速的神奇魔法

當我們呼叫 ChatGPT 時，字是一個個噴出來的（自迴歸生成）。
當預測下一個 Token 時：
$$x_{new} = f(\text{prompt} + \text{generated\_tokens})$$
如果我們每次生成新字，都要重新計算 Prompt 裡所有字的注意力，計算複雜度會隨著長度呈二次方 $O(L^2)$ 增長。

**KV Cache** 的思想是：**過去計算過的 Key 和 Value 是不會變的，只有當前新生成的 Token 的 $K_t, V_t$ 需要被計算。**
因此，我們將歷史的 $K$ 和 $V$ 儲存起來，每次只需計算當前 Token 的 $Q_t, K_t, V_t$，並將 $K_t, V_t$ 拼接（Concat）到快取中。
這直接將單步解碼的時間複雜度從 $O(L^2)$ 降到了 $O(L)$。

```
Step 1: 輸入 "I love"  ──► 計算其 K_0, V_0 ──► 儲存到 KV Cache
Step 2: 生成 "AI"      ──► 只計算 "AI" 的 K_1, V_1 ──► 與 Cache 拼接 ──► 預測下一個字
```

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
