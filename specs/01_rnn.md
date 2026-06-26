# 規格說明：循環神經網路 (RNN & LSTM)

本規格書定義了在 PyTorch 中實作自訂 RNN 與 LSTM Cell 的行為、數學公式、張量維度（Tensor Shapes）以及驗證指引。

---

## 1. 模組概述
循環神經網路（RNN）藉由維護一個在時間步之間傳遞的隱藏狀態 $h_t$，來對序列資料進行建模。

---

## 2. 數學定義

### 2.1 標準 RNN Cell
對於序列中的每一個時間步，隱藏狀態 $h_t$ 更新如下：
$$h_t = \tanh(X_t W_{ih}^T + b_{ih} + h_{t-1} W_{hh}^T + b_{hh})$$

### 2.2 LSTM Cell
對於序列中的每一個時間步，細胞狀態 $C_t$ 和隱藏狀態 $h_t$ 經由四個控制門更新：
1.  **遺忘門 ($f_t$)**：決定要丟棄多少舊有的細胞狀態。
    $$f_t = \sigma(X_t W_{if}^T + b_{if} + h_{t-1} W_{hf}^T + b_{hf})$$
2.  **輸入門 ($i_t$)**：決定要寫入多少新的資訊。
    $$i_t = \sigma(X_t W_{ii}^T + b_{ii} + h_{t-1} W_{hi}^T + b_{hi})$$
3.  **候選細胞狀態 ($\tilde{C}_t$)**：準備寫入的新記憶候選值。
    $$\tilde{C}_t = \tanh(X_t W_{ig}^T + b_{ig} + h_{t-1} W_{hg}^T + b_{hg})$$
4.  **細胞狀態更新 ($C_t$)**：
    $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
5.  **輸出門 ($o_t$)**：決定要輸出哪些部分的隱藏狀態。
    $$o_t = \sigma(X_t W_{io}^T + b_{io} + h_{t-1} W_{ho}^T + b_{ho})$$
6.  **隱藏狀態更新 ($h_t$)**：
    $$h_t = o_t \odot \tanh(C_t)$$

其中：
*   $\sigma$ 代表 Sigmoid 激活函數。
*   $\odot$ 代表按元素相乘（Hadamard Product）。

---

## 3. 張量維度 (Tensor Shapes)

| 變數 / 參數 | 維度 (Shape) | 描述 |
| :--- | :--- | :--- |
| `x` (目前步輸入) | `(B, D_in)` | Batch 大小為 $B$，輸入特徵維度為 $D_{in}$ 的當前輸入 |
| `h` (隱藏狀態) | `(B, D_hid)` | 來自前一步的隱藏狀態 |
| `c` (細胞狀態，僅 LSTM) | `(B, D_hid)` | 來自前一步的細胞狀態 |
| `W_ih` / `W_ix` | `(D_hid, D_in)` | 輸入到隱藏層的權重 |
| `W_hh` | `(D_hid, D_hid)` | 循環隱藏層權重 |
| `b_ih` / `b_hh` | `(D_hid,)` | 偏置 (Biases) |

---

## 4. 驗證要求 (單元測試)

為確保實作正確（遵循 Superpowers 測試驗證流程）：
1.  **維度驗證**：確保輸出形狀與 `(B, D_hid)` 完全一致。
2.  **與 PyTorch 等價性測試**：
    *   初始化您的自訂 Cell。
    *   以相同的權重與偏置初始化 PyTorch 原生的 `nn.RNNCell` 與 `nn.LSTMCell`。
    *   比較前向傳播的輸出結果，最大絕對誤差需 $< 1\times 10^{-6}$。
3.  **梯度流動測試**：對輸出求和並進行反向傳播，確保所有參數都能接收到非零梯度。
