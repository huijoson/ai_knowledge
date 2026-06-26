# Specification: Recurrent Neural Networks (RNN & LSTM)

This specification defines the behavior, mathematical formulas, tensor shapes, and verification guidelines for implementing custom RNN and LSTM cells in PyTorch.

---

## 1. Module Overview
Recurrent Neural Networks model sequential data by maintaining a hidden state $h_t$ that carries information from previous steps.

---

## 2. Mathematical Definition

### 2.1 Standard RNN Cell
For each step in the sequence, the hidden state $h_t$ is updated as:
$$h_t = \tanh(X_t W_{ih}^T + b_{ih} + h_{t-1} W_{hh}^T + b_{hh})$$

### 2.2 LSTM Cell
For each step in the sequence, the cell state $C_t$ and hidden state $h_t$ are updated using four gates:
1.  **Forget Gate**: controls how much of the past cell state to keep.
    $$f_t = \sigma(X_t W_{if}^T + b_{if} + h_{t-1} W_{hf}^T + b_{hf})$$
2.  **Input Gate**: controls how much new information to add.
    $$i_t = \sigma(X_t W_{ii}^T + b_{ii} + h_{t-1} W_{hi}^T + b_{hi})$$
3.  **Candidate Cell State**: the proposed new values.
    $$\tilde{C}_t = \tanh(X_t W_{ig}^T + b_{ig} + h_{t-1} W_{hg}^T + b_{hg})$$
4.  **Cell State Update**:
    $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
5.  **Output Gate**: controls what to output.
    $$o_t = \sigma(X_t W_{io}^T + b_{io} + h_{t-1} W_{ho}^T + b_{ho})$$
6.  **Hidden State Update**:
    $$h_t = o_t \odot \tanh(C_t)$$

Where:
*   $\sigma$ is the Sigmoid activation function.
*   $\odot$ is the element-wise (Hadamard) product.

---

## 3. Tensor Shapes

| Variable / Parameter | Shape | Description |
| :--- | :--- | :--- |
| `x` (Input step) | `(B, D_in)` | Current input at step $t$ for batch size $B$ and input dimension $D_{in}$ |
| `h` (Hidden state) | `(B, D_hid)` | Hidden state from step $t-1$ |
| `c` (Cell state, LSTM only) | `(B, D_hid)` | Cell state from step $t-1$ |
| `W_ih` / `W_ix` | `(D_hid, D_in)` | Input-to-hidden weights |
| `W_hh` | `(D_hid, D_hid)` | Hidden-to-hidden weights |
| `b_ih` / `b_hh` | `(D_hid,)` | Biases |

---

## 4. Verification Requirements (Tests)

To verify the implementation (adhering to Superpowers workflow):
1.  **Shape Verification**: Ensure outputs match `(B, D_hid)`.
2.  **Equivalence with PyTorch**:
    *   Initialize your custom cells.
    *   Initialize PyTorch's native `nn.RNNCell` and `nn.LSTMCell` with the exact same weights and biases.
    *   Compare forward pass outputs; maximum absolute difference should be $< 1\times 10^{-6}$.
3.  **Gradient Flow**: Perform a backward pass on a loss term and ensure all parameters receive non-zero gradients (no dead gradient paths).
