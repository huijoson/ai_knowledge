# AI Knowledge Learning Sandbox

This repository is dedicated to learning core artificial intelligence concepts by implementing foundational neural network architectures from scratch.

---

## 📬 Repository Structure

*   **`specs/`**: Specification sheets using **OpenSpec** definitions. These files detail the mathematics, tensor shapes, and operational flow before coding.
*   **`src/`**: Implementation folder for each module.
    *   `src/rnn/`: Standard RNN and LSTM Cell implementations.
    *   `src/cnn/`: 2D Convolution and Pooling layers.
    *   `src/attention/`: Scaled Dot-Product and Multi-Head Attention.
    *   `src/transformer/`: Encoder, Decoder, and sequence-to-sequence Transformer.
*   **`tests/`**: Unit tests modeled around the **Superpowers** verification workflow to mathematically prove the correctness of the code.

---

## 🛠️ Getting Started

### 1. Install Dependencies
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Running Verification Tests
To run unit tests (which verify implementation shapes and mathematical equivalence with PyTorch modules):
```bash
pytest
```
*Initially, tests for unimplemented modules will skip or raise `NotImplementedError`.*

---

## 📚 Learning Journey

1.  **Module 1: RNN**
    *   Read [specs/01_rnn.md](file:///Users/yuhan/coding/ai_knowledge/specs/01_rnn.md) to understand gates and recurrence.
    *   Implement [CustomRNNCell](file:///Users/yuhan/coding/ai_knowledge/src/rnn/rnn_cell.py) and [CustomLSTMCell](file:///Users/yuhan/coding/ai_knowledge/src/rnn/lstm_cell.py).
    *   Run `pytest tests/test_rnn.py` to verify.
2.  **Module 2: CNN**
    *   Read [specs/02_cnn.md](file:///Users/yuhan/coding/ai_knowledge/specs/02_cnn.md).
3.  **Module 3: Self-Attention**
    *   Read [specs/03_attention.md](file:///Users/yuhan/coding/ai_knowledge/specs/03_attention.md).
4.  **Module 4: The Transformer**
    *   Read [specs/04_transformer.md](file:///Users/yuhan/coding/ai_knowledge/specs/04_transformer.md).
