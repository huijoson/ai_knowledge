import torch
import torch.nn as nn
from .attention import MultiHeadAttention, GroupedQueryAttention
from .normalization import CustomLayerNorm, CustomRMSNorm
from .feed_forward import VanillaFFN, SwiGLUFFN
from .positional_encoding import SinusoidalPositionalEncoding, RotaryPositionEmbedding

class Seq2SeqTransformer(nn.Module):
    """
    Vanilla Encoder-Decoder Seq2Seq Transformer model.
    """
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int, num_heads: int, num_layers: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model)
        
        # Skeletons for stack components:
        # For simplicity, we can use PyTorch standard Transformer modules,
        # but the goal is to implement them from scratch.
        # This wrapper will be completed by the student/agent during coding sessions.
        
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            src (Tensor): (B, L_src) source token IDs
            tgt (Tensor): (B, L_tgt) target token IDs
        """
        # TODO: Implement full encoder-decoder forward pass
        raise NotImplementedError("Implement Seq2SeqTransformer forward pass!")


class DecoderOnlyLLM(nn.Module):
    """
    Modern Decoder-Only Transformer architecture (e.g., LLaMA, GPT).
    Features:
        - RMSNorm (Pre-normalization)
        - RoPE Positional Embedding
        - SwiGLU Feed-Forward Network
        - Grouped-Query Attention (GQA)
    """
    def __init__(self, vocab_size: int, d_model: int, num_query_heads: int, num_kv_heads: int, num_layers: int, d_ff: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        
        # RMSNorm before output layer
        self.norm = CustomRMSNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens (Tensor): (B, L) token IDs
        """
        # TODO: Implement modern Decoder-only forward pass.
        # 1. Look up embeddings.
        # 2. Loop through DecoderLayers (using GQA + RoPE + RMSNorm + SwiGLU).
        # 3. Apply final RMSNorm.
        # 4. Project to vocabulary logits.
        raise NotImplementedError("Implement DecoderOnlyLLM forward pass!")
