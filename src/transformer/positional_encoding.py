import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding from the original Transformer.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # Create a static positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as a buffer so it moves to GPU with the model but is not trained
        self.register_buffer('pe', pe.unsqueeze(0)) # Shape: (1, max_len, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): shape (B, L, d_model)
        """
        # Add the static encoding to the input embedding
        # Slice pe to match sequence length L
        return x + self.pe[:, :x.size(1), :]


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) used in LLaMA models.
    It rotates Query and Key representations to encode relative positions.
    """
    def __init__(self, dim: int, max_seq_len: int = 4096):
        super().__init__()
        self.dim = dim
        
        # Calculate theta and frequency matrix
        # theta_i = 10000^(-2(i-1)/d)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqss = torch.outer(t, self.inv_freq) # (max_seq_len, dim // 2)
        
        # [cos(theta*m), sin(theta*m)]
        freqs_cos = torch.cos(freqss)
        freqs_sin = torch.sin(freqss)
        
        self.register_buffer("freqs_cos", freqs_cos, persistent=False) # (max_seq_len, dim // 2)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False) # (max_seq_len, dim // 2)
        
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        # Rotate half of the channels in x: [-x2, x1]
        x1 = x[..., :self.dim // 2]
        x2 = x[..., self.dim // 2:]
        return torch.cat((-x2, x1), dim=-1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): shape (B, H, L, d_k) where H is head count, d_k is head dim (d_k == dim)
        Returns:
            Tensor: RoPE rotated tensor of shape (B, H, L, d_k)
        """
        # TODO: Apply RoPE rotation to query or key tensor.
        # Formula: x_rotated = x * cos(m*theta) + rotate_half(x) * sin(m*theta)
        # Note: Broad-cast cos and sin to match (B, H, L, d_k)
        raise NotImplementedError("Implement RoPE forward pass!")
