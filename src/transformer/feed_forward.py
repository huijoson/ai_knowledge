import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaFFN(nn.Module):
    """
    Position-wise Feed-Forward Network using ReLU.
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): shape (B, L, d_model)
        """
        # TODO: Implement w_2(ReLU(w_1(x)))
        raise NotImplementedError("Implement VanillaFFN forward pass!")


class SwiGLUFFN(nn.Module):
    """
    SwiGLU (Swish Gated Linear Unit) Feed-Forward Network.
    Formula:
        FFN_SwiGLU(x) = (Swish(x * W_gate) * (x * W_up)) * W_down
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # Usually d_ff for SwiGLU is calculated as 2/3 of 4 * d_model, rounded to multiple of 256.
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): shape (B, L, d_model)
        """
        # TODO: Implement SwiGLU forward pass.
        # Hint: F.silu is PyTorch's native Swish activation.
        raise NotImplementedError("Implement SwiGLUFFN forward pass!")
