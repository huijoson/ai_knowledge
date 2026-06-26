import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Attention (MHA) as proposed in Vaswani et al., 2017.
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            q (Tensor): Query tensor of shape (B, L_q, d_model)
            k (Tensor): Key tensor of shape (B, L_k, d_model)
            v (Tensor): Value tensor of shape (B, L_k, d_model)
            mask (Tensor): Optional attention mask of shape (B, 1, L_q, L_k) or broad-castable
            
        Returns:
            Tensor: Output context vectors of shape (B, L_q, d_model)
        """
        # TODO: Implement multi-head split, scaled dot-product attention, merge heads, and output projection.
        raise NotImplementedError("Implement MultiHeadAttention forward pass!")


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) used in modern LLMs like LLaMA 3.
    """
    def __init__(self, d_model: int, num_query_heads: int, num_kv_heads: int):
        super().__init__()
        assert num_query_heads % num_kv_heads == 0, "Query heads must be divisible by KV heads"
        
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_query_heads
        self.num_queries_per_kv = num_query_heads // num_kv_heads
        
        self.W_q = nn.Linear(d_model, num_query_heads * self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(num_query_heads * self.d_k, d_model, bias=False)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, L, d_model)
            mask (Tensor): Attention mask
            
        Returns:
            Tensor: Output tensor of shape (B, L, d_model)
        """
        # TODO: Project x to q, k, v. Repeat/broadcast k and v heads to match q heads, perform attention, and project back.
        raise NotImplementedError("Implement GroupedQueryAttention forward pass!")
