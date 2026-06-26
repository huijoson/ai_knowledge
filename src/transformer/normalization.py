import torch
import torch.nn as nn

class CustomLayerNorm(nn.Module):
    """
    Standard Layer Normalization layer with learnable scale (gamma) and shift (beta).
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): shape (..., d_model)
        """
        # TODO: Calculate mean and variance across the last dimension, normalize, and apply gamma/beta.
        raise NotImplementedError("Implement CustomLayerNorm forward pass!")


class CustomRMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm) used in LLaMA models.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): shape (..., d_model)
        """
        # TODO: Calculate RMS scaling and apply it to x along with learnable parameter gamma.
        raise NotImplementedError("Implement CustomRMSNorm forward pass!")
