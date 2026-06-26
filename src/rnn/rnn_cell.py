import torch
import torch.nn as nn

class CustomRNNCell(nn.Module):
    """
    A custom implementation of a standard Elman RNN cell.
    
    Formula:
        h_t = tanh(x_t * W_ih^T + b_ih + h_prev * W_hh^T + b_hh)
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights and biases
        # W_ih shape: (hidden_size, input_size)
        self.W_ih = nn.Parameter(torch.Tensor(hidden_size, input_size))
        # W_hh shape: (hidden_size, hidden_size)
        self.W_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        
        self.b_ih = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hh = nn.Parameter(torch.Tensor(hidden_size))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize weights using standard uniform bounds (like PyTorch does)."""
        import math
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): input tensor of shape (B, D_in)
            h_prev (Tensor): previous hidden state of shape (B, D_hid)
            
        Returns:
            Tensor: next hidden state of shape (B, D_hid)
        """
        # TODO: Implement the standard RNN cell step calculation here.
        # Hint: Use torch.tanh, torch.matmul (or @), and transpose/transpose-like shapes as defined in specs/01_rnn.md.
        raise NotImplementedError("Implement the RNN cell forward pass!")
