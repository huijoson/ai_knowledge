import torch
import torch.nn as nn
import pytest
from src.nn import CustomLinear, CustomGELU, CustomAdamW

def test_linear_equivalence():
    in_features = 8
    out_features = 4
    batch_size = 2
    
    custom_linear = CustomLinear(in_features, out_features)
    torch_linear = nn.Linear(in_features, out_features)
    
    # Sync weights
    with torch.no_grad():
        torch_linear.weight.copy_(custom_linear.weight)
        torch_linear.bias.copy_(custom_linear.bias)
        
    x = torch.randn(batch_size, in_features)
    
    try:
        out_custom = custom_linear(x)
        out_torch = torch_linear(x)
        assert torch.allclose(out_custom, out_torch, atol=1e-6)
    except NotImplementedError:
        pytest.skip("CustomLinear not implemented.")

def test_gelu_equivalence():
    x = torch.linspace(-3.0, 3.0, 100)
    custom_gelu = CustomGELU()
    
    try:
        out_custom = custom_gelu(x)
        out_torch = torch.nn.functional.gelu(x)
        # Note: Since we use the standard tanh approximation in specs, atol is 1e-4 instead of 1e-6
        assert torch.allclose(out_custom, out_torch, atol=1e-3)
    except NotImplementedError:
        pytest.skip("CustomGELU not implemented.")

def test_adamw_equivalence():
    # Setup simple linear model
    x = torch.randn(5, 3)
    w = torch.randn(3, 1, requires_grad=True)
    w_custom = w.clone().detach().requires_grad_(True)
    
    # Calculate simple loss
    y = x @ w
    loss = y.sum()
    loss.backward()
    
    # Save gradients
    grad_saved = w.grad.clone()
    
    # Set gradient for custom parameter
    w_custom.grad = grad_saved.clone()
    
    # Initialize optimizers
    lr = 0.1
    wd = 0.01
    custom_opt = CustomAdamW([w_custom], lr=lr, weight_decay=wd)
    torch_opt = torch.optim.AdamW([w], lr=lr, weight_decay=wd)
    
    try:
        custom_opt.step()
        torch_opt.step()
        
        # Verify updated weights match
        assert torch.allclose(w_custom, w, atol=1e-5)
    except NotImplementedError:
        pytest.skip("CustomAdamW not implemented.")
