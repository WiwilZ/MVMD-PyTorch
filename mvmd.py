import math

import torch


def mvmd(signal: torch.Tensor, K: int, alpha=2000, tau=0.0, DC=False, init=1, tol=1e-7, N=500):
    """
    Multivariate Variational Mode Decomposition implemented in PyTorch.

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
    if signal.dim() != 2:
        raise ValueError("signal must be a 2D tensor of shape (T, C)")
    T, C = signal.size()
    dtype, device = signal.dtype, signal.device
    if dtype is torch.float32:
        ctype = torch.complex64
    elif dtype is torch.float64:
        ctype = torch.complex128
    else:
        raise NotImplementedError(f"dtype {dtype} not supported")

    """Preparations"""
    # Mirroring
    pad = T // 2
    left = signal[:pad].flip(dims=(0,))           
    right = signal[-pad:].flip(dims=(0,))        
    f = torch.cat([left, signal, right], dim=0) 
    L = f.size(0)  # 2T if T is even else 2T - 1 
    freqs = torch.linspace(0, 1 - 1 / L, L, dtype=dtype, device=device) - 0.5
    # Construct and center f_hat
    f_hat_plus = torch.fft.fftshift(torch.fft.fft(f, dim=0), dim=0)  # (L, C)
    f_hat_plus[:L // 2] = 0

    """Initialization"""
    # matrix keeping track of every iterant
    u_hat_plus = torch.zeros(L, C, K, dtype=ctype, device=device)
    u_hat_prev = u_hat_plus.clone()
    omega_plus = torch.empty(N, K, dtype=dtype, device=device)
    # initialize omegas uniformly
    if init == 1:
        omega_plus[0] = (0.5 / K) * torch.arange(K, device=device)
    elif init == 2:
        omega_plus[0] = (math.log(0.5 * T) * torch.rand(K, device=device).sort()[0]).exp() / T
    else:
        omega_plus[0] = 0
    # if DC mode imposed, set its omega to 0
    if DC:
        omega_plus[0, 0] = 0

    """Algorithm of MVMD"""
    u_diff = tol + torch.finfo(float).eps
    n = 0
    lambda_hat = torch.zeros(L, C, dtype=ctype, device=device)
    sum_uk = torch.zeros(L, C, dtype=ctype, device=device)
    while u_diff > tol and n < N - 1:
        # update first mode
        sum_uk += u_hat_prev[:, :, -1] - u_hat_prev[:, :, 0]
        u_hat_plus[:, :, 0] = (f_hat_plus - sum_uk - lambda_hat / 2) / (1 + alpha * (freqs.unsqueeze(1) - omega_plus[n, 0]).square())
        if not DC:
            power = u_hat_plus[L // 2:, :, 0].abs().square()
            omega_plus[n + 1, 0] = (freqs[L // 2:].unsqueeze(1) * power).sum() / (power.sum() + torch.finfo(dtype).eps)
        # update other modes
        for k in range(1, K):
            sum_uk += u_hat_plus[:, :, k - 1] - u_hat_prev[:, :, k]
            u_hat_plus[:, :, k] = (f_hat_plus - sum_uk - lambda_hat / 2) / (1 + alpha * (freqs.unsqueeze(1) - omega_plus[n, k]).square())
            power = u_hat_plus[L // 2:, :, k].abs().square()
            omega_plus[n + 1, k] = (freqs[L // 2:].unsqueeze(1) * power).sum() / (power.sum() + torch.finfo(dtype).eps)
        # Dual ascent
        lambda_hat += tau * (u_hat_plus.sum(dim=2) - f_hat_plus)
        # convergence
        u_diff = (u_hat_plus - u_hat_prev).abs().square().sum().item() / L + torch.finfo(float).eps
        u_hat_prev = u_hat_plus.clone()
        n += 1

    """Post-processing and cleanup"""
    # discard empty space if converged early
    omega = omega_plus[:n]
    # Signal reconstruction
    u_hat = torch.empty_like(u_hat_plus)
    u_hat[L // 2:] = u_hat_plus[L // 2:]
    u_hat[1:L // 2 + 1] = u_hat_plus[L // 2:].conj().flip(dims=(0,))
    u_hat[0] = u_hat[-1].conj()

    u = torch.fft.ifft(torch.fft.ifftshift(u_hat, dim=0), dim=0).real
    # remove mirror part 
    u = u[pad:-pad]
    # recompute spectrum
    u_hat = torch.fft.fftshift(torch.fft.fft(u, dim=0), dim=0)

    return u, u_hat, omega


if __name__ == "__main__":
    T = 1000
    t = torch.linspace(1 / T, 1, T, dtype=float)
    f_channel1 = torch.cos(2*torch.pi*2*t) + 1/16*torch.cos(2*torch.pi*36*t)
    f_channel2 = 1/4*torch.cos(2*torch.pi*24*t) + 1/16*torch.cos(2*torch.pi*36*t)
    signal = torch.stack([f_channel1, f_channel2], dim=1)
    u, u_hat, omega = mvmd(signal, 3)
