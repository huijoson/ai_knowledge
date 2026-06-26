import torch
import torch.nn as nn

class CustomLSTMCell(nn.Module):
    """
    A custom implementation of a Long Short-Term Memory (LSTM) cell.
    
    Refer to specs/01_rnn.md for the mathematical equations of forget, input, output, and candidate gates.
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # We can implement the gates separately or combine them into single projection matrices for efficiency.
        # Let's define individual weights for clarity of learning:
        # Gates: input (i), forget (f), cell candidate (g), output (o)
        
        self.W_ii = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ii = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hi = nn.Parameter(torch.Tensor(hidden_size))
        
        self.W_if = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_if = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hf = nn.Parameter(torch.Tensor(hidden_size))
        
        self.W_ig = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ig = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hg = nn.Parameter(torch.Tensor(hidden_size))
        
        self.W_io = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_io = nn.Parameter(torch.Tensor(hidden_size))
        self.b_ho = nn.Parameter(torch.Tensor(hidden_size))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        import math
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (Tensor): input tensor of shape (B, D_in)
            states (tuple of Tensors): (h_prev, c_prev) each of shape (B, D_hid)
            
        Returns:
            tuple of Tensors: (h_next, c_next) each of shape (B, D_hid)
        """
        h_prev, c_prev = states
        # TODO: Implement the LSTM cell equations using torch.sigmoid, torch.tanh, and matrix multiplications.
        # Refer to specs/01_rnn.md for details.
        raise NotImplementedError("Implement the LSTM cell forward pass!")
