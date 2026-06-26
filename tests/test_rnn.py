import torch
import torch.nn as nn
import pytest
from src.rnn import CustomRNNCell, CustomLSTMCell

def test_rnn_cell_forward_shape():
    batch_size = 4
    input_size = 8
    hidden_size = 16
    
    cell = CustomRNNCell(input_size, hidden_size)
    x = torch.randn(batch_size, input_size)
    h_prev = torch.randn(batch_size, hidden_size)
    
    try:
        h_next = cell(x, h_prev)
        assert h_next.shape == (batch_size, hidden_size)
    except NotImplementedError:
        pytest.skip("CustomRNNCell forward pass not yet implemented.")

def test_rnn_cell_equivalence():
    batch_size = 2
    input_size = 4
    hidden_size = 8
    
    custom_cell = CustomRNNCell(input_size, hidden_size)
    torch_cell = nn.RNNCell(input_size, hidden_size)
    
    # Load same weights/biases
    with torch.no_grad():
        torch_cell.weight_ih.copy_(custom_cell.W_ih)
        torch_cell.weight_hh.copy_(custom_cell.W_hh)
        torch_cell.bias_ih.copy_(custom_cell.b_ih)
        torch_cell.bias_hh.copy_(custom_cell.b_hh)
        
    x = torch.randn(batch_size, input_size)
    h_prev = torch.randn(batch_size, hidden_size)
    
    try:
        h_next_custom = custom_cell(x, h_prev)
        h_next_torch = torch_cell(x, h_prev)
        
        # Verify output similarity
        assert torch.allclose(h_next_custom, h_next_torch, atol=1e-6)
    except NotImplementedError:
        pytest.skip("CustomRNNCell forward pass not yet implemented.")

def test_lstm_cell_equivalence():
    batch_size = 2
    input_size = 4
    hidden_size = 8
    
    custom_cell = CustomLSTMCell(input_size, hidden_size)
    torch_cell = nn.LSTMCell(input_size, hidden_size)
    
    # Copy weights
    with torch.no_grad():
        # In PyTorch LSTMCell, weight_ih has shape (4*hidden, input)
        # weight_hh has shape (4*hidden, hidden)
        # Splitting PyTorch weight order is: input gate, forget gate, cell gate, output gate (i, f, g, o)
        torch_cell.weight_ih.copy_(torch.cat([
            custom_cell.W_ii,
            custom_cell.W_if,
            custom_cell.W_ig,
            custom_cell.W_io
        ], dim=0))
        
        torch_cell.weight_hh.copy_(torch.cat([
            custom_cell.W_hi,
            custom_cell.W_hf,
            custom_cell.W_hg,
            custom_cell.W_ho
        ], dim=0))
        
        torch_cell.bias_ih.copy_(torch.cat([
            custom_cell.b_ii,
            custom_cell.b_if,
            custom_cell.b_ig,
            custom_cell.b_io
        ], dim=0))
        
        torch_cell.bias_hh.copy_(torch.cat([
            custom_cell.b_hi,
            custom_cell.b_hf,
            custom_cell.b_hg,
            custom_cell.b_ho
        ], dim=0))
        
    x = torch.randn(batch_size, input_size)
    h_prev = torch.randn(batch_size, hidden_size)
    c_prev = torch.randn(batch_size, hidden_size)
    
    try:
        h_custom, c_custom = custom_cell(x, (h_prev, c_prev))
        h_torch, c_torch = torch_cell(x, (h_prev, c_prev))
        
        assert torch.allclose(h_custom, h_torch, atol=1e-6)
        assert torch.allclose(c_custom, c_torch, atol=1e-6)
    except NotImplementedError:
        pytest.skip("CustomLSTMCell forward pass not yet implemented.")
