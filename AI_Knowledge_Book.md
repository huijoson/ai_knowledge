# 從零開始：自手實作深度學習與大語言模型核心架構 (終極學習與實作指南)

歡迎閱讀這本書！本書專為希望藉由「從零開始手寫實作（From Scratch）」來真正吃透現代深度學習與大語言模型（LLM）底層機制的研究者與工程師所寫。

我們拒絕僅停留在理論公式，也拒絕直接呼叫高階套件。我們將結合 **OpenSpec（規格驅動）** 與 **Superpowers（測試驅動）** 的軟體工程邏輯，把數學原理一步步轉化為能跑、可過單元測試的 PyTorch 程式碼。

---

## 📖 本書大綱

*   [導讀：零數學直覺導引 — 用生活白話解碼硬核數學](#導讀零數學直覺導引--用生活白話解碼硬核數學)
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

## 導讀：零數學直覺導引 — 用生活白話解碼硬核數學

如果您看到數學公式就頭痛，請別擔心！深度學習的數學符號本質上只是**用來精確描述生活常識的簡寫**。

在進入硬核公式前，我們先用「白話翻譯官」把這些數學符號與概念翻譯成生活常識：

### 0.1 數學符號白話翻譯對照表

| 數學符號 | 英文名稱 | 生活白話解釋 (直覺比喻) | Python 程式碼對照 |
| :--- | :--- | :--- | :--- |
| $\sum_{i=1}^N x_i$ | Summation | **「全部加起來」**。把一堆數字（如帳單明細）全部加總。 | `sum(x)` |
| $\odot$ | Hadamard Product | **「疊加濾鏡」**。兩張一樣大的照片，對應的像素點互相相乘（如 Photoshop 的正片疊底）。 | `a * b` |
| $\sigma(x)$ | Sigmoid | **「開關閥門」**。把任何數字壓縮到 $0$ (關閉) 到 $1$ (全開) 之間，用來表示比例或機率。 | `torch.sigmoid(x)` |
| $\nabla_\theta L$ | Gradient | **「下坡最陡的方向」**。指引你往哪個方向走，誤差（Loss）會降得最快。 | `theta.grad` |
| $W^T$ | Transpose | **「橫豎對調」**。把一個表格的列和行交換，以便和別的表格對齊做計算。 | `W.T` or `W.t()` |
| $A \cdot B$ (或 $AB$) | Matrix Multiply | **「多重條件篩選」**。把多個輸入和多組權重進行配對配方計算。 | `A @ B` or `torch.matmul(A, B)` |

---

### 0.2 八大核心概念的生活比喻

#### 1. 計算圖與自動求導 (Autograd) $\rightarrow$ 「拼圖的組裝與倒帶推演」
*   **前向傳播**：就像照著說明書組裝一輛樂高玩具車。你把一個個小零件（輸入）組裝成輪子，再拼成底盤，最後得到整台車（輸出）。
*   **反向傳播**：如果最後車子跑偏了（有誤差），我們需要「倒帶」回去檢查。我們會問：輪子歪了影響了底盤多少？底盤歪了又影響了整台車多少？這種**「環環相扣的倒帶責任追究」**，在數學上就叫**鏈式法則**。

#### 2. 線性層與激活函數 $\rightarrow$ 「配方調配與過濾門檻」
*   **線性層 ($Y = XW^T + b$)**：就像是做調酒。不同的原料（輸入 $X$）乘上不同的比例配方（權重 $W$），再加上一點基底（偏置 $b$），調出新的風味（輸出 $Y$）。
*   **激活函數 (ReLU/GELU)**：就像是**「篩選門檻」**。如果調出來的酒酒精濃度低於某個值，我們直接倒掉（ReLU 歸零）；如果高於門檻，才讓它通過。

#### 3. 優化器 (AdamW) $\rightarrow$ 「在濃霧中滾下山的小球」
*   你要在濃霧籠罩的山頂走到山谷最低點（尋找最小誤差）。
*   **SGD**：每走一步，就低頭看腳下哪裡最陡，就往那裡跨一步。容易卡在小水坑。
*   **AdamW**：給小球加上**「慣性」**（動量，遇到小水坑能衝過去）與**「自適應剎車」**（如果某個方向一直很陡，就走大步點；如果方向一直變，就走小步點），並且在每次更新時，小球會自動**「磨損變小」**（Decoupled Weight Decay，防止權重過大）。

#### 4. 循環神經網路 (RNN & LSTM) $\rightarrow$ 「看電影的記憶筆記」
*   **傳統 RNN**：你看電影時，不帶筆記本，只靠腦袋記。看到第 2 小時，你已經把前 10 分鐘的細節忘光了（梯度消失）。
*   **LSTM**：你帶了一本**「精美筆記本（Cell State）」**，並配備了三隻彩色筆：
    *   *遺忘門*：用橡皮擦擦掉過期的無用資訊（如：前一幕已經死掉的角色）。
    *   *輸入門*：用紅筆寫下剛出現的重要伏筆。
    *   *輸出門*：用螢光筆畫出當前這一幕需要注意的重點。

#### 5. 卷積神經網路 (CNN) $\rightarrow$ 「拿放大鏡找線索的偵探」
*   偵探不需要一次看完整張大地圖。他拿著一個**「放大鏡（Kernel）」**在照片上從左到右、從上到下一格格掃描，只尋找特定特徵（如：貓耳朵、車輪）。不論線索出現在地圖的哪裡，放大鏡都能捕捉到（平移不變性）。

#### 6. 自注意力機制 (Attention) $\rightarrow$ 「圖書館的條碼檢索系統」
*   你要在圖書館找書。
*   **Query (Q)**：你想找的主題（例如：「我想找關於蘋果派的食譜」）。
*   **Key (K)**：書架上所有書的「封面標籤/條碼」（如：一本寫著「甜點大全」，另一本寫著「世界歷史」）。
*   **Value (V)**：書本內「實際的內容知識」。
*   系統會計算你的 **Query** 和所有 **Key** 的相似度（點積），算出配對權重（Softmax），然後根據權重把相關書籍的**實際內容 (Value)** 調取出來給你。

#### 7. 旋轉位置編碼 (RoPE) $\rightarrow$ 「指北針與相對旋轉」
*   傳統編碼像是在每本書封面上貼上絕對頁碼（「第 3 頁」）。
*   **RoPE** 則是給向量裝上**「旋轉指北針」**。隨著位置增加，向量會像時鐘指針一樣旋轉一個角度。兩個向量指針之間的**「夾角」**，就直接代表了它們之間的相對距離。

#### 8. KV Cache $\rightarrow$ 「開卷考試的草稿紙」
*   當老師一題題問你 Prompt 後續的答案時，你不需要每次都把前面的題目和已寫好的答案重新看一遍、算一遍。
*   你把算過的步驟寫在**「草稿紙（KV Cache）」**上。每次回答新問題，只需看前一題寫了什麼，並把新答案記在草稿紙末端，避免重算，速度提升百倍。

---

## 第一章：自動求導 (Autograd) 與計算圖 — 深度學習的物理心臟

在 PyTorch 等框架中，我們只需要寫好前向計算（Forward），反向求導（Backward）就會自動完成。這背後的物理心臟就是「動態計算圖」。

### 1.1 計算圖的數學原理
任何複雜的函數都可以分解為一系列基礎運算（加、減、乘、除、矩陣相乘等）。

我們以函數 $z = (x + y) \times w$ 為例。
1.  **定義輸入與中間節點**：
    *   輸入：$x = 2$, $y = 3$, $w = 4$
    *   加法節點：$u = x + y = 2 + 3 = 5$
    *   乘法節點：$z = u \times w = 5 \times 4 = 20$
2.  **鏈式法則的反向流動**：
    當我們想要知道「$z$ 的變化受 $x$ 的影響有多大」（即偏微分 $\frac{\partial z}{\partial x}$）時，我們從輸出端 $z$ 逆向往回推算：
    *   首先計算 $\frac{\partial z}{\partial u}$。因為 $z = u \times w$，所以 $\frac{\partial z}{\partial u} = w = 4$。
    *   接著計算 $\frac{\partial u}{\partial x}$。因為 $u = x + y$，所以 $\frac{\partial u}{\partial x} = 1$。
    *   根據鏈式法則，將兩者相乘：
        $$\frac{\partial z}{\partial x} = \frac{\partial z}{\partial u} \cdot \frac{\partial u}{\partial x} = 4 \times 1 = 4$$

---

### 1.2 🧐 手動算一遍 (Autograd 實戰)
設我們有以下輸入值：
*   $x = 3.0$
*   $y = 5.0$
*   $w = 2.0$

**前向傳播（計算數值）**：
1.  $u = x + y = 3.0 + 5.0 = 8.0$
2.  $z = u \times w = 8.0 \times 2.0 = 16.0$

**反向傳播（計算梯度）**：
1.  起點梯度 $\frac{\partial z}{\partial z} = 1.0$。
2.  對權重 $w$ 的梯度：$\frac{\partial z}{\partial w} = u = 8.0$。
3.  對中間變數 $u$ 的梯度：$\frac{\partial z}{\partial u} = w = 2.0$。
4.  對輸入 $x$ 的梯度：$\frac{\partial z}{\partial x} = \frac{\partial z}{\partial u} \cdot \frac{\partial u}{\partial x} = 2.0 \times 1.0 = 2.0$。
5.  對輸入 $y$ 的梯度：$\frac{\partial z}{\partial y} = \frac{\partial z}{\partial u} \cdot \frac{\partial u}{\partial y} = 2.0 \times 1.0 = 2.0$。

#### 💻 對應的 PyTorch 程式碼驗證
```python
import torch

# requires_grad=True 告訴 PyTorch 建立這個節點的計算圖
x = torch.tensor(3.0, requires_grad=True)
y = torch.tensor(5.0, requires_grad=True)
w = torch.tensor(2.0, requires_grad=True)

# 前向傳播
u = x + y
z = u * w

# 反向傳播
z.backward()

# 印出結果
print(x.grad) # 輸出: tensor(2.0)
print(y.grad) # 輸出: tensor(2.0)
print(w.grad) # 輸出: tensor(8.0)
```

### 1.3 互動式自動求導與計算圖視覺化工具
您可以調整下方滑動條改變輸入與權重，並點選「前向傳播」與「反向傳播」觀察數據與梯度的傳遞路徑與粒子流向動畫。

<div id="autograd-visualizer" class="interactive-visualizer"></div>

---

## 第二章：基礎層與激活函數 (GELU) — 讓網路具備非線性表達能力

### 2.1 線性投影層 (Linear Layer) 的矩陣計算
線性層的作用是將輸入特徵 $X$ 乘以權重矩陣 $W$ 並加上偏置 $b$。
$$Y = X W^T + b$$

#### 📐 維度拆解說明
假設我們輸入一個 batch 有 2 個句子，每個句子用 3 維的向量表示，我們要將其投影到 4 維的空間：
*   **輸入 $X$**：維度為 `(2, 3)` (Batch_Size=2, In_Features=3)
*   **權重 $W$**：維度為 `(4, 3)` (Out_Features=4, In_Features=3)
*   **偏置 $b$**：維度為 `(4)`
*   **計算 $X W^T$**：
    *   $W^T$ (轉置後) 的維度為 `(3, 4)`。
    *   矩陣乘法 `(2, 3) @ (3, 4)` $\rightarrow$ 得到的輸出 $Y$ 維度為 `(2, 4)`。

---

### 2.2 🧐 手動算一遍 (線性層矩陣乘法)
設：
$$X = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}, \quad W = \begin{pmatrix} 0.5 & 1.5 \\ -1.0 & 2.0 \end{pmatrix}, \quad b = \begin{pmatrix} 0.1 \\ -0.2 \end{pmatrix}$$

計算 $Y = X W^T + b$：
1.  **先算 $W^T$ (轉置)**：
    $$W^T = \begin{pmatrix} 0.5 & -1.0 \\ 1.5 & 2.0 \end{pmatrix}$$
2.  **進行矩陣相乘 $X W^T$**：
    *   第一列第一行：$1 \times 0.5 + 2 \times 1.5 = 0.5 + 3.0 = 3.5$
    *   第一列第二行：$1 \times (-1.0) + 2 \times 2.0 = -1.0 + 4.0 = 3.0$
    *   第二列第一行：$3 \times 0.5 + 4 \times 1.5 = 1.5 + 6.0 = 7.5$
    *   第二列第二行：$3 \times (-1.0) + 4 \times 2.0 = -3.0 + 8.0 = 5.0$
    $$X W^T = \begin{pmatrix} 3.5 & 3.0 \\ 7.5 & 5.0 \end{pmatrix}$$
3.  **加上偏置 $b$** (廣播加到每一列上)：
    $$Y = \begin{pmatrix} 3.5 + 0.1 & 3.0 - 0.2 \\ 7.5 + 0.1 & 5.0 - 0.2 \end{pmatrix} = \begin{pmatrix} 3.6 & 2.8 \\ 7.6 & 4.8 \end{pmatrix}$$

#### 💻 自訂 Linear 層與 PyTorch 實現對照
您可以在 [linear.py](file:///Users/yuhan/coding/ai_knowledge/src/nn/linear.py) 中實作以下 forward 計算：
```python
# 手寫自訂實作
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # 這裡的 self.weight 維度是 (out_features, in_features)
    # 使用 t() 進行轉置，或者使用 nn.functional.linear
    return torch.matmul(x, self.weight.t()) + self.bias
```

---

### 2.3 GELU 激活函數計算細節
GELU 激活函數將輸入乘以高斯累積分布機率：
$$\text{GELU}(x) \approx 0.5x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \left(x + 0.044715 x^3\right)\right)\right)$$

#### 📐 變數說明
*   $\pi \approx 3.14159$
*   $\sqrt{2/\pi} \approx 0.79788$

#### 💻 GELU 實作代碼
您可以在 [activation.py](file:///Users/yuhan/coding/ai_knowledge/src/nn/activation.py) 中實作以下結構：
```python
import torch
import torch.nn as nn
import math

class CustomGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 近似公式實作
        const = math.sqrt(2.0 / math.pi)
        inner = const * (x + 0.044715 * torch.pow(x, 3))
        return 0.5 * x * (1.0 + torch.tanh(inner))
```

---

## 第三章：優化器的藝術 (AdamW) — 訓練大語言模型的黃金鑰匙

AdamW 優化器的核心思想是為每個參數動態調整學習率，同時對參數進行與梯度無關的直接權重衰減。

### 3.1 AdamW 演算法的關鍵公式拆解
在每個時間步 $t$，對每個參數 $\theta$：
1.  **更新一階動量 (動量)**：$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
    *   *含意*：累積過去的梯度方向，防範當前 Batch 雜訊。
2.  **更新二階動量 (自適應分母)**：$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
    *   *含意*：估計每個參數的波動程度，用來做自適應縮放。
3.  **偏差修正**：
    $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
    *   *含意*：因為第 $1$ 步的 $m_0, v_0$ 為 0，前幾步計算出來的 $m_t, v_t$ 會太接近 0，需要除以修正項將其放大。
4.  **Decoupled 權重衰減與更新**：
    $$\theta_{t+1} = \theta_t - \eta \cdot \lambda \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
    *   *含意*：$\eta \cdot \lambda \theta_t$ 代表將當前權重直接縮小一點（L2 正則化），後半段則是 Adam 的標準更新。

---

### 3.2 💻 AdamW 的自訂 step 實作
您可以在 [optimizer.py](file:///Users/yuhan/coding/ai_knowledge/src/nn/optimizer.py) 中實作如下核心步驟：
```python
# 取得對應狀態
state = self.state[p]
exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
step = state['step']

# 1. 權重衰減直接套用 (Decoupled Weight Decay)
p.mul_(1.0 - lr * wd)

# 2. 更新矩估計
exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

# 3. 計算偏差修正分母
bias_correction1 = 1.0 - beta1 ** step
bias_correction2 = 1.0 - beta2 ** step

# 4. 套用自適應步長更新參數
step_size = lr / bias_correction1
denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

p.addcdiv_(exp_avg, denom, value=-step_size)
```

---

## 第四章：循環神經網路 (RNN & LSTM) — 時間與序列的建模

### 4.1 LSTM 門控更新公式與維度對照
LSTM 維護兩個狀態：
*   $C_t$ (Cell State)：細胞狀態，長短期記憶載體。
*   $h_t$ (Hidden State)：隱藏狀態，當前步輸出。

#### 📐 各門控權重與維度拆解
在 LSTMCell 中，我們要為四個門分別計算輸入與隱藏狀態的投影：
*   忘記門 $f_t$、輸入門 $i_t$、候選門 $g_t$、輸出門 $o_t$。
*   對於每個門，我們有權重矩陣 $W_{i*} \in \mathbb{R}^{D_{hid} \times D_{in}}$ 與 $W_{h*} \in \mathbb{R}^{D_{hid} \times D_{hid}}$。

---

### 4.2 💻 LSTMCell 實作對照
您可以在 [lstm_cell.py](file:///Users/yuhan/coding/ai_knowledge/src/rnn/lstm_cell.py) 中，將四個門的投影計算以 Python 實作出來：
```python
def forward(self, x: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    h_prev, c_prev = states
    
    # 1. 忘記門 (Forget Gate)
    f_t = torch.sigmoid(F.linear(x, self.W_if, self.b_if) + F.linear(h_prev, self.W_hf, self.b_hf))
    
    # 2. 輸入門 (Input Gate)
    i_t = torch.sigmoid(F.linear(x, self.W_ii, self.b_ii) + F.linear(h_prev, self.W_hi, self.b_hi))
    
    # 3. 候選細胞狀態 (Candidate Cell State)
    g_t = torch.tanh(F.linear(x, self.W_ig, self.b_ig) + F.linear(h_prev, self.W_hg, self.b_hg))
    
    # 4. 更新細胞狀態 (Cell State Update)
    c_next = f_t * c_prev + i_t * g_t
    
    # 5. 輸出門 (Output Gate)
    o_t = torch.sigmoid(F.linear(x, self.W_io, self.b_io) + F.linear(h_prev, self.W_ho, self.b_ho))
    
    # 6. 更新隱藏狀態 (Hidden State Update)
    h_next = o_t * torch.tanh(c_next)
    
    return h_next, c_next
```

---

## 第五章：卷積神經網路 (CNN) — 空間特徵提取與下採樣

### 5.1 2D 卷積滑動視窗計算公式與輸出維度
當我們用一個核心大小為 $K \times K$、步長為 $S$、填充為 $P$ 的卷積核處理輸入時，輸出空間解析度的計算公式如下：

$$H_{out} = \left\lfloor \frac{H_{in} - K_h + 2P_h}{S_h} \right\rfloor + 1$$

---

### 5.2 🧐 手動算一遍 (輸出維度計算)
設：
*   輸入 Image 高度 $H_{in} = 5$
*   Kernel 高度 $K_h = 3$
*   Padding $P_h = 1$
*   Stride $S_h = 2$

將數據帶入公式：
$$H_{out} = \left\lfloor \frac{5 - 3 + 2 \times 1}{2} \right\rfloor + 1 = \left\lfloor \frac{4}{2} \right\rfloor + 1 = 2 + 1 = 3$$
代表卷積層輸出特徵圖的高度將會是 $3$。

### 5.3 互動式 2D 卷積滑動核視覺化工具
點擊下方的播放按鈕，觀察卷積核如何在 5x5 的圖像上滑動，計算出 3x3 的特徵圖結果。

<div id="cnn-visualizer" class="interactive-visualizer"></div>

---

## 第六章：注意力機制 (Attention) — 從 MHA 到大模型的 GQA 與 MQA

自注意力機制的核心在於計算序列內部各 Token 的相似度得分。

### 6.1 Scaled Dot-Product Attention 的手動算一遍
假設我們的輸入序列只有 2 個 Token，每個 Token 的特徵維度 $d_k = 4$。
設投影後的 Query 矩陣 $Q$ 和 Key 矩陣 $K$ 分別為：
$$Q = \begin{pmatrix} 1.0 & 0.0 & 1.0 & 0.0 \\ 0.0 & 2.0 & 0.0 & 1.0 \end{pmatrix}, \quad K = \begin{pmatrix} 1.0 & 0.0 & 1.0 & 0.0 \\ 0.0 & 1.0 & 0.0 & 1.0 \end{pmatrix}$$

1.  **計算點積矩陣 $Q K^T$**：
    *   首先計算 $K^T$：
        $$K^T = \begin{pmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \\ 1.0 & 0.0 \\ 0.0 & 1.0 \end{pmatrix}$$
    *   計算相乘 $Q K^T$：
        $$QK^T = \begin{pmatrix} 1\times1 + 1\times1 & 0 \\ 0 & 2\times1 + 1\times1 \end{pmatrix} = \begin{pmatrix} 2.0 & 0.0 \\ 0.0 & 3.0 \end{pmatrix}$$
2.  **除以縮放因子 $\sqrt{d_k}$**：
    *   因為 $d_k = 4$，所以 $\sqrt{d_k} = 2.0$。
    *   縮放後得分：
        $$\text{Scaled Scores} = \frac{QK^T}{2.0} = \begin{pmatrix} 1.0 & 0.0 \\ 0.0 & 1.5 \end{pmatrix}$$
3.  **套用 Softmax 得到注意力權重 (對每一列)**：
    *   對於第一列 `[1.0, 0.0]`：
        $$\text{softmax}([1.0, 0.0]) = \left[ \frac{e^1}{e^1 + e^0}, \frac{e^0}{e^1 + e^0} \right] \approx \left[ \frac{2.718}{3.718}, \frac{1}{3.718} \right] \approx [0.73, 0.27]$$
    *   第一列 Token 對於位置 0 分配了 73% 的注意力，對於位置 1 分配了 27% 的注意力。

---

### 6.2 💻 多頭注意力中的 Split 與 Merge 實作
您可以在 [attention.py](file:///Users/yuhan/coding/ai_knowledge/src/transformer/attention.py) 中，使用 `view` 和 `transpose` 來處理多頭分拆：
```python
def forward(self, q, k, v, mask=None):
    batch_size, seq_len, d_model = q.shape
    
    # 1. 線性投影並拆分多頭
    # Shape transition: (B, L, D) -> (B, L, H, d_k) -> (B, H, L, d_k)
    q_heads = self.W_q(q).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    k_heads = self.W_k(k).view(batch_size, k.size(1), self.num_heads, self.d_k).transpose(1, 2)
    v_heads = self.W_v(v).view(batch_size, k.size(1), self.num_heads, self.d_k).transpose(1, 2)
    
    # 2. 計算 Scaled Dot-Product Attention
    scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / math.sqrt(self.d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == float('-inf'), float('-inf'))
    attn_weights = torch.softmax(scores, dim=-1)
    context = torch.matmul(attn_weights, v_heads)
    
    # 3. 合併多頭 (Merge Heads)
    # Shape transition: (B, H, L, d_k) -> (B, L, H, d_k) -> (B, L, D)
    context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
    return self.W_o(context)
```

### 6.3 互動式自注意力矩陣權重視覺化工具
將滑鼠懸停在不同的單字（Query）上，觀察自注意力機制如何為各個 Token 分配不同的權重強度（以顏色深淺表示）。

<div id="attention-visualizer" class="interactive-visualizer"></div>

---

## 第七章：現代大語言模型核心組件 — RMSNorm, SwiGLU 與 RoPE

### 7.1 RMSNorm 實作邏輯與公式對照
RMSNorm 相比 LayerNorm 去除了減去均值步驟，公式如下：

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}$$

#### 💻 RMSNorm 實作代碼
您可以在 [normalization.py](file:///Users/yuhan/coding/ai_knowledge/src/transformer/normalization.py) 中寫出以下高效張量運算：
```python
class CustomRMSNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 沿著最後一維計算平方平均值
        variance = x.pow(2).mean(-1, keepdim=True)
        # 2. 乘以 rsqrt (1 / sqrt(variance + eps))
        x_normed = x * torch.rsqrt(variance + self.eps)
        # 3. 套用學得的縮放尺度 gamma (self.gamma)
        return x_normed * self.gamma
```

---

### 7.2 SwiGLU 實作與公式對照
SwiGLU 將前饋網路（FFN）雙分支相乘，並以 Swish (SiLU) 當作門控：
$$\text{SwiGLU}(x) = (\text{Swish}(x W_{gate}) \odot x W_{up}) W_{down}$$

#### 💻 SwiGLU 實作代碼
您可以在 [feed_forward.py](file:///Users/yuhan/coding/ai_knowledge/src/transformer/feed_forward.py) 中實作如下：
```python
class SwiGLUFFN(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.silu 就是 PyTorch 內建的 Swish 函數
        gate_branch = F.silu(self.w_gate(x))
        up_branch = self.w_up(x)
        # 點對點相乘 (Hadamard Product) 後投射回原空間
        return self.w_down(gate_branch * up_branch)
```

---

## 第八章：LLM 推理效能優化 (KV Cache) — 推理加速的神奇魔法

自迴歸解碼生成新字時，我們需要快取過去計算好的 Key 和 Value。

### 8.1 KV Cache 的維度變化與拼接
在推理的單步解碼中：
*   輸入的序列長度僅為 $1$（即上一步剛剛產生的那個單字），輸入維度為 `(B, 1, d_model)`。
*   經過 $W_K, W_V$ 投影後得到當前步的 Key 和 Value，$K_{new}, V_{new}$ 的維度為 `(B, H_kv, 1, d_k)`。
*   假設歷史快取 $K_{cached}, V_{cached}$ 的維度是 `(B, H_kv, L_prev, d_k)`。
*   我們將其沿著**序列維度 (Sequence Dimension, dim=2)** 進行拼接：
    $$K_{cached} \leftarrow \text{cat}([K_{cached}, K_{new}], \text{dim}=2) \quad \text{(維度變為: (B, H_kv, L_prev + 1, d_k))}$$
    $$V_{cached} \leftarrow \text{cat}([V_{cached}, V_{new}], \text{dim}=2) \quad \text{(維度變為: (B, H_kv, L_prev + 1, d_k))}$$

#### 💻 KV Cache 更新虛擬碼
```python
# 每次輸入只有一個 Token (seq_len=1)
q_new, k_new, v_new = project(x) # k_new 尺寸為 (B, H, 1, d_k)

# 拼接至歷史快取中
k_cache = torch.cat([k_cache, k_new], dim=2)
v_cache = torch.cat([v_cache, v_new], dim=2)

# 注意力點積計算：Query長度是1，Key長度是 L_prev + 1
scores = torch.matmul(q_new, k_cache.transpose(-2, -1)) # 輸出尺寸 (B, H, 1, L_prev + 1)
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
