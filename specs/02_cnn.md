# Specification: Convolutional Neural Networks (CNN)

This specification defines the behavior, mathematical formulas, tensor shapes, and verification guidelines for implementing a custom 2D Convolution layer and 2D Max Pooling layer from scratch.

---

## 1. Module Overview
Convolutional Neural Networks are designed for spatial data, utilizing sliding kernels to extract features locally while sharing weights across the spatial domain.

---

## 2. Mathematical Definition

### 2.1 2D Convolution (Cross-Correlation)
For a single output element at position $(i, j)$ in output channel $c_{out}$ and batch index $b$:
$$Y_{b, c_{out}, i, j} = b_{c_{out}} + \sum_{c_{in}=0}^{C_{in}-1} \sum_{m=0}^{K_h-1} \sum_{n=0}^{K_w-1} X_{b, c_{in}, i \cdot S + m, j \cdot S + n} \odot W_{c_{out}, c_{in}, m, n}$$

Where:
*   $X$ is the padded input tensor.
*   $W$ is the kernel weights.
*   $b$ is the bias.
*   $S$ is the stride.
*   $K_h, K_w$ are the height and width of the kernel.

### 2.2 Output Spatial Dimensions
Given input height $H_{in}$ and width $W_{in}$, kernel size $(K_h, K_w)$, padding $(P_h, P_w)$, and stride $(S_h, S_w)$, the output dimensions $(H_{out}, W_{out})$ are:
$$H_{out} = \left\lfloor \frac{H_{in} - K_h + 2P_h}{S_h} \right\rfloor + 1$$
$$W_{out} = \left\lfloor \frac{W_{in} - K_w + 2P_w}{S_w} \right\rfloor + 1$$

### 2.3 2D Max Pooling
For a pooling window of size $(K_h, K_w)$ and stride $S$:
$$Y_{b, c, i, j} = \max_{m=0 \dots K_h-1, n=0 \dots K_w-1} X_{b, c, i \cdot S + m, j \cdot S + n}$$

---

## 3. Tensor Shapes

| Variable / Parameter | Shape | Description |
| :--- | :--- | :--- |
| `x` (Input tensor) | `(B, C_in, H_in, W_in)` | Batch size $B$, Input channels $C_{in}$, height, width |
| `W` (Kernel weights) | `(C_out, C_in, K_h, K_w)`| Output channels, Input channels, kernel height, kernel width |
| `bias` | `(C_out,)` | Bias per output channel |
| `y` (Output tensor) | `(B, C_out, H_out, W_out)`| Output activation map |

---

## 4. Verification Requirements (Tests)

To verify the implementation:
1.  **Padding Correctness**: Ensure edge padding (constant 0) works correctly before performing convolution.
2.  **Output Dimensions**: Test various kernel sizes, strides, and padding sizes to verify the output shape matches the theoretical formula.
3.  **Equivalence with PyTorch**:
    *   Initialize custom conv/pooling layers and load weights/biases into them.
    *   Initialize PyTorch's native `nn.Conv2d` and `nn.MaxPool2d` with the exact same weights and parameters.
    *   Verify that outputs match within a tolerance of $< 1\times 10^{-6}$.
4.  **Backward Pass Validation**:
    *   Ensure backward gradients with respect to input $X$ and weights $W$ match PyTorch's autograd gradients.
