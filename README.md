# mvmd-pytorch
Multivariate Variational Mode Decomposition implemented in PyTorch

Reference: https://www.mathworks.com/matlabcentral/fileexchange/72814-multivariate-variational-mode-decomposition-mvmd

# API
```python
def mvmd(signal: torch.Tensor, K: int, alpha=2000.0, tau=0.0, DC=False, init=1, tol=1e-7, N=500) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Multivariate Variational Mode Decomposition implemented in PyTorch.

    Args:
        signal (torch.Tensor): Input tensor of shape (B, T, C) or (T, C) where B is batch size, T is signal length and C is channels.
        K               (int): Number of modes to extract.
        alpha         (float): Bandwidth constraint parameter.
        tau           (float): Time-step for dual ascent (noise slack).
        DC             (bool): If True, enforce first mode at DC (zero frequency).
        init            (int): Initialization type for omegas (1=uniform, 2=random, else zeros).
        tol           (float): Convergence tolerance for ADMM.
        N               (int): Maximum number of iterations.

    Returns:
        u     (torch.Tensor): Reconstructed modes of shape (B, T, C, K) or (T, C, K).
        u_hat (torch.Tensor): Spectra of modes, shape (B, T, C, K) or (T, C, K).
        omega (torch.Tensor): Center frequencies per iteration, shape (B, n, K) or (n, K).
    """
```

# Usage
```python
B, T = 2, 1000
t = torch.linspace(1 / T, 1, T, dtype=float)
f_channel1 = torch.cos(2*torch.pi*2*t) + 1/16*torch.cos(2*torch.pi*36*t)
f_channel2 = 1/4*torch.cos(2*torch.pi*24*t) + 1/16*torch.cos(2*torch.pi*36*t)
signal = torch.stack([f_channel1, f_channel2], dim=1)
u, u_hat, omega = mvmd(signal.unsqueeze(0).expand(B, *signal.shape), 3)
```
