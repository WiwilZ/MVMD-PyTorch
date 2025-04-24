# mvmd-pytorch
Multivariate Variational Mode Decomposition implemented in PyTorch

Reference: https://www.mathworks.com/matlabcentral/fileexchange/72814-multivariate-variational-mode-decomposition-mvmd

# Usage
```python
def mvmd(signal: torch.Tensor, K: int, alpha=2000, tau=0.0, DC=False, init=1, tol=1e-7, N=500):
    """
    Args:
        signal (torch.Tensor): Input tensor of shape (T, C) where T is signal length and C is channels.
        alpha (float): Bandwidth constraint parameter.
        tau (float): Time-step for dual ascent (noise slack).
        K (int): Number of modes to extract.
        DC (bool): If True, enforce first mode at DC (zero frequency).
        init (int): Initialization type for omegas (1=uniform, 2=random, other=zeros).
        tol (float): Convergence tolerance for ADMM.
        N (int): Maximum number of iterations.
    
    Returns:
        u (torch.Tensor): Reconstructed modes of shape (T, C, K).
        u_hat (torch.Tensor): Spectra of modes, shape (T, C, K).
        omega (torch.Tensor): Center frequencies per iteration, shape (n, K).
    """
```
