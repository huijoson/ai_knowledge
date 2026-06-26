import torch
import torch.nn as nn
import math

class CustomLinear(nn.Module):
    """
    A custom implementation of a Linear (Fully-Connected) Layer.
    Calculates: Y = X * W^T + b
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Define weights and biases
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.in_features) if self.in_features > 0 else 0
        nn.init.uniform_(self.weight, -stdv, stdv)
        nn.init.uniform_(self.bias, -stdv, stdv)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (..., in_features)
        Returns:
            Tensor: Output tensor of shape (..., out_features)
        """
        # TODO: Implement the linear transformation here.
        # Hint: Use torch.matmul (or @) and handle dimensions correctly.
        raise NotImplementedError("Implement CustomLinear forward pass!")
