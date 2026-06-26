# 規格說明：卷積神經網路 (CNN)

本規格書定義了從零開始實作自訂 2D 卷積層（Convolution）與 2D 最大池化層（Max Pooling）的行為、數學公式、張量維度（Tensor Shapes）以及驗證指引。

---

## 1. 模組概述
卷積神經網路專為空間資料（如影像）設計，利用滑動核（Kernels）提取局部特徵，並在整個空間域共享權重。

---

## 2. 數學定義

### 2.1 2D 卷積 (互相關運算)
對於輸出通道 $c_{out}$、批次索引 $b$ 在空間位置 $(i, j)$ 的單個輸出元素：
$$Y_{b, c_{out}, i, j} = b_{c_{out}} + \sum_{c_{in}=0}^{C_{in}-1} \sum_{m=0}^{K_h-1} \sum_{n=0}^{K_w-1} X_{b, c_{in}, i \cdot S + m, j \cdot S + n} \odot W_{c_{out}, c_{in}, m, n}$$

其中：
*   $X$ 是進行了邊緣填充（Padding）後的輸入張量。
*   $W$ 是卷積核權重。
*   $b$ 是偏置。
*   $S$ 是步長（Stride）。
*   $K_h, K_w$ 是卷積核的高度與寬度。

### 2.2 輸出空間維度計算
給定輸入高度 $H_{in}$ 與寬度 $W_{in}$、卷積核大小 $(K_h, K_w)$、填充 $(P_h, P_w)$ 以及步長 $(S_h, S_w)$，輸出維度 $(H_{out}, W_{out})$ 為：
$$H_{out} = \left\lfloor \frac{H_{in} - K_h + 2P_h}{S_h} \right\rfloor + 1$$
$$W_{out} = \left\lfloor \frac{W_{in} - K_w + 2P_w}{S_w} \right\rfloor + 1$$

### 2.3 2D 最大池化 (Max Pooling)
對於大小為 $(K_h, K_w)$、步長為 $S$ 的池化窗口：
$$Y_{b, c, i, j} = \max_{m=0 \dots K_h-1, n=0 \dots K_w-1} X_{b, c, i \cdot S + m, j \cdot S + n}$$

---

## 3. 張量維度 (Tensor Shapes)

| 變數 / 參數 | 維度 (Shape) | 描述 |
| :--- | :--- | :--- |
| `x` (輸入張量) | `(B, C_in, H_in, W_in)` | 批次大小 $B$，輸入通道數 $C_{in}$，高度，寬度 |
| `W` (卷積核權重) | `(C_out, C_in, K_h, K_w)`| 輸出通道數，輸入通道數，核心高，核心寬 |
| `bias` | `(C_out,)` | 每個輸出通道的偏置 |
| `y` (輸出張量) | `(B, C_out, H_out, W_out)`| 輸出特徵圖 (Activation Map) |

---

## 4. 驗證要求 (單元測試)

為確保實作正確：
1.  **邊緣填充正確性**：在進行卷積之前，確保邊緣填充（常數 0 填充）能正確運作。
2.  **輸出維度驗證**：測試不同的卷積核大小、步長和填充大小，確保輸出的 shape 符合理論公式。
3.  **與 PyTorch 等價性測試**：
    *   初始化您的自訂卷積/池化層，並將權重與偏置載入其中。
    *   以相同的權重與參數初始化 PyTorch 原生的 `nn.Conv2d` 與 `nn.MaxPool2d`。
    *   驗證兩者的輸出結果，誤差需 $< 1\times 10^{-6}$。
4.  **反向傳播驗證**：
    *   確保對輸入 $X$ 和權重 $W$ 的梯度與 PyTorch autograd 的計算結果一致。
