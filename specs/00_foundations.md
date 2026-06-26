# 規格說明：深度學習基礎與優化器 (Foundations & Optimizers)

本規格書定義了深度學習最核心的基礎組件：計算圖（Computational Graph）與自動求導（Autograd）原理、基礎層（線性層、激活函數、損失函數）以及主流優化器（SGD、AdamW）。

---

## 1. 模組概述
任何深度學習框架的底層都是「自動求導引擎」。我們必須理解前向傳播（Forward Pass）建立計算圖，以及反向傳播（Backward Pass）利用鏈式法則（Chain Rule）傳遞梯度的過程。同時，優化器利用這些梯度更新參數，使模型收斂。

---

## 2. 數學定義

### 2.1 線性層 (Linear / Fully-Connected Layer)
$$Y = X W^T + b$$
*   $X \in \mathbb{R}^{B \times D_{in}}$：輸入張量。
*   $W \in \mathbb{R}^{D_{out} \times D_{in}}$：權重矩陣。
*   $b \in \mathbb{R}^{D_{out}}$：偏置。

---

### 2.2 核心激活函數 (Activations)

#### 1. ReLU (Rectified Linear Unit)
$$\text{ReLU}(x) = \max(0, x)$$
梯度：當 $x > 0$ 時為 1，否則為 0。

#### 2. GELU (Gaussian Error Linear Unit)
BERT 與 GPT 採用的標準激活函數。它藉由輸入的值來加權決定要隨機丟棄或保留資訊：
$$\text{GELU}(x) = x \cdot \Phi(x) \approx 0.5x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \left(x + 0.044715 x^3\right)\right)\right)$$

---

### 2.3 損失函數 (Loss Functions)

#### 1. MSE Loss (Mean Squared Error)
主要用於回歸任務：
$$\text{MSE}(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$

#### 2. 交叉熵損失 (Cross Entropy Loss)
主要用於分類與語言模型預測下一個 Token 任務：
$$\text{CELoss}(y, \hat{y}) = -\sum_{i=1}^C y_i \log(\text{softmax}(\hat{y}_i))$$

---

### 2.4 優化器 (Optimizers)

#### 1. SGD (Stochastic Gradient Descent)
$$\theta_{t+1} = \theta_t - \eta \cdot g_t$$
其中 $\eta$ 為學習率，$g_t$ 為梯度。

#### 2. AdamW (Adam with Decoupled Weight Decay)
現代大模型訓練的黃金標準優化器。它將權重衰減（L2 正則化）與動量（Momentum）和自適應學習率（RMSProp）解耦：
1.  **一階動量（梯度均值）**：$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
2.  **二階動量（梯度未中心化的變異數）**：$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
3.  **偏差修正**：$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$, $\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$
4.  **參數更新（含 decoupled weight decay）**：
    $$\theta_{t+1} = \theta_t - \eta \cdot \lambda \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
    其中 $\lambda$ 為權重衰減率。

---

## 3. 驗證要求 (單元測試)

為確保實作正確：
1.  **線性層反向傳播**：實作一個簡單的線性層前向與手寫反向傳播，與 PyTorch 的 `grad` 進行比對，確保對 $W$ 與 $b$ 的偏微分正確。
2.  **GELU 激活函數精確度**：比對自訂 GELU 與 `torch.nn.functional.gelu`，確保數值誤差 $< 1\times 10^{-5}$。
3.  **AdamW 更新邏輯**：給定固定梯度，手動模擬 AdamW 更新一步，並與 PyTorch 的 `torch.optim.AdamW` 輸出進行比對。
