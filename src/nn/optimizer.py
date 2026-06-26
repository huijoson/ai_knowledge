import torch
from torch.optim import Optimizer

class CustomAdamW(Optimizer):
    """
    Custom implementation of the AdamW optimizer (Adam with decoupled weight decay).
    
    Refer to specs/00_foundations.md for mathematical equations of first-order momentum (m),
    second-order momentum (v), bias corrections, and decoupled weight decay updates.
    """
    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), 
                 eps: float = 1e-8, weight_decay: float = 1e-2):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                
                # Retrieve state for this parameter
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                step = state['step']
                
                # TODO: Implement the AdamW updates:
                # 1. Apply decoupled weight decay directly to parameter: p = p - lr * wd * p
                # 2. Update exp_avg (m_t) and exp_avg_sq (v_t) using betas.
                # 3. Calculate bias corrections: bias_correction1 = 1 - beta1^step, bias_correction2 = 1 - beta2^step
                # 4. Compute adapted learning rate step size and update parameter weights.
                
                raise NotImplementedError("Implement CustomAdamW step logic!")
                
        return loss
