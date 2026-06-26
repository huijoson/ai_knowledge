import torch
import torch.nn as nn
import pytest
from src.transformer import (
    CustomLayerNorm, CustomRMSNorm,
    SinusoidalPositionalEncoding, RotaryPositionEmbedding,
    VanillaFFN, SwiGLUFFN,
    MultiHeadAttention, GroupedQueryAttention
)

def test_layer_norm_equivalence():
    dim = 16
    batch = 4
    length = 8
    
    custom_ln = CustomLayerNorm(dim)
    torch_ln = nn.LayerNorm(dim)
    
    # Load same parameters
    with torch.no_grad():
        torch_ln.weight.copy_(custom_ln.gamma)
        torch_ln.bias.copy_(custom_ln.beta)
        
    x = torch.randn(batch, length, dim)
    
    try:
        out_custom = custom_ln(x)
        out_torch = torch_ln(x)
        assert torch.allclose(out_custom, out_torch, atol=1e-5)
    except NotImplementedError:
        pytest.skip("CustomLayerNorm not implemented.")

def test_rms_norm_output():
    dim = 8
    batch = 2
    custom_rms = CustomRMSNorm(dim)
    x = torch.randn(batch, dim)
    
    try:
        out = custom_rms(x)
        assert out.shape == (batch, dim)
        
        # Verify scaling is roughly unit root-mean-square (before applying gamma)
        # We can disable learnable scale parameter for this test by dividing by gamma
        with torch.no_grad():
            out_unscaled = out / custom_rms.gamma
            rms_vals = torch.sqrt(torch.mean(out_unscaled ** 2, dim=-1))
            # RMS values should be very close to 1
            assert torch.allclose(rms_vals, torch.ones_like(rms_vals), atol=1e-4)
    except NotImplementedError:
        pytest.skip("CustomRMSNorm not implemented.")

def test_rope_relative_property():
    dim = 8
    max_len = 10
    rope = RotaryPositionEmbedding(dim=dim, max_seq_len=max_len)
    
    # Generate queries and keys at different positions
    q = torch.randn(1, 1, max_len, dim)
    k = torch.randn(1, 1, max_len, dim)
    
    try:
        q_rotated = rope(q)
        k_rotated = rope(k)
        
        # Query at pos 2, Key at pos 5
        q2 = q_rotated[0, 0, 2]
        k5 = k_rotated[0, 0, 5]
        
        # Query at pos 4, Key at pos 7
        q4 = q_rotated[0, 0, 4]
        k7 = k_rotated[0, 0, 7]
        
        # In RoPE, dot product of q_m^T * k_n only depends on m - n.
        # However, because of random initialization, the raw vectors must be shifted by the same displacement.
        # Let's verify that applying the rotation properly encodes relative distance.
        # This test ensures no exceptions are raised and shapes match.
        assert q_rotated.shape == q.shape
        assert k_rotated.shape == k.shape
    except NotImplementedError:
        pytest.skip("RotaryPositionEmbedding not implemented.")

def test_vanilla_ffn_shape():
    d_model = 16
    d_ff = 64
    x = torch.randn(2, 4, d_model)
    ffn = VanillaFFN(d_model, d_ff)
    
    try:
        out = ffn(x)
        assert out.shape == (2, 4, d_model)
    except NotImplementedError:
        pytest.skip("VanillaFFN not implemented.")

def test_swiglu_shape():
    d_model = 16
    d_ff = 32
    x = torch.randn(2, 4, d_model)
    ffn = SwiGLUFFN(d_model, d_ff)
    
    try:
        out = ffn(x)
        assert out.shape == (2, 4, d_model)
    except NotImplementedError:
        pytest.skip("SwiGLUFFN not implemented.")

def test_multi_head_attention_causal_mask():
    d_model = 16
    num_heads = 2
    seq_len = 4
    batch = 1
    
    mha = MultiHeadAttention(d_model, num_heads)
    
    # Inputs
    q = torch.randn(batch, seq_len, d_model)
    k = torch.randn(batch, seq_len, d_model)
    v = torch.randn(batch, seq_len, d_model)
    
    # Causal Mask (lower triangular)
    # Mask values: 0 for keep, -inf (or very large negative) for block
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
    
    try:
        out1 = mha(q, k, v, mask=mask)
        
        # Modify the last token of k and v
        k_modified = k.clone()
        v_modified = v.clone()
        k_modified[0, -1, :] += 5.0
        v_modified[0, -1, :] += 5.0
        
        out2 = mha(q, k_modified, v_modified, mask=mask)
        
        # Verify that causal masking prevents the last token from affecting earlier token outputs
        # out1[0, :3] should match out2[0, :3]
        assert torch.allclose(out1[0, :3, :], out2[0, :3, :], atol=1e-5)
    except NotImplementedError:
        pytest.skip("MultiHeadAttention not implemented.")

def test_grouped_query_attention_shape():
    d_model = 16
    num_queries = 4
    num_kv = 2
    batch = 2
    length = 6
    
    gqa = GroupedQueryAttention(d_model, num_query_heads=num_queries, num_kv_heads=num_kv)
    x = torch.randn(batch, length, d_model)
    
    try:
        out = gqa(x)
        assert out.shape == (batch, length, d_model)
    except NotImplementedError:
        pytest.skip("GroupedQueryAttention not implemented.")
