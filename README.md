# AI 核心架構手手實作沙盒

本專案旨在透過「從零開始（From Scratch）」動手實作核心神經網路架構，來學習與理解人工智慧的核心知識。

---

## 📬 專案目錄結構

*   **`specs/`**：使用 **OpenSpec** 規範編寫的模組規格書，詳細定義了寫程式前的數學公式、張量維度（Tensor Shapes）與運算流程。
*   **`src/`**：各單元的實作原始碼。
    *   `src/rnn/`：標準 RNNCell 與 LSTMCell 實作。
    *   `src/transformer/`：多頭注意力（MHA）、分組查詢注意力（GQA）、RMSNorm、SwiGLU、RoPE 以及完整的 Transformer 與 Decoder-only LLM 模型。
*   **`tests/`**：基於 **Superpowers** 驗證流程設計的單元測試，用以數學性地證明實作程式碼的正確性。
*   **`index.html`**：本地與線上部署的互動式電子書閱讀器。

---

## 🛠️ 開發環境安裝

### 1. 安裝依賴套件
確保您已安裝 Python，並在專案根目錄下執行：
```bash
pip install -r requirements.txt
```

### 2. 執行單元測試驗證
若要驗證自訂模組是否與 PyTorch 官方高度優化的 C++ 核心計算結果完全等價，請執行：
```bash
pytest
```
*初始狀態下，未實作的模組測試會自動跳過（Skipped）或拋出 `NotImplementedError`。*

---

## 📚 學習指南

1.  **第一單元：循環神經網路 (RNN & LSTM)**
    *   閱讀 [specs/01_rnn.md](file:///Users/yuhan/coding/ai_knowledge/specs/01_rnn.md) 以理解門控與狀態傳遞。
    *   實作 [CustomRNNCell](file:///Users/yuhan/coding/ai_knowledge/src/rnn/rnn_cell.py) 與 [CustomLSTMCell](file:///Users/yuhan/coding/ai_knowledge/src/rnn/lstm_cell.py)。
    *   執行 `pytest tests/test_rnn.py` 驗證。
2.  **第二單元：卷積神經網路 (CNN)**
    *   閱讀 [specs/02_cnn.md](file:///Users/yuhan/coding/ai_knowledge/specs/02_cnn.md)。
3.  **第三單元：Transformer 與現代 LLM**
    *   閱讀 [specs/03_transformer.md](file:///Users/yuhan/coding/ai_knowledge/specs/03_transformer.md)。
    *   實作 [src/transformer/](file:///Users/yuhan/coding/ai_knowledge/src/transformer/) 目錄下的注意力、歸一化與前饋模組。
    *   執行 `pytest tests/test_transformer.py` 驗證。
