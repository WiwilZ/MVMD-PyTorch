import math

import torch


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
    if signal.dim() == 2:
        signal = signal.unsqueeze(0)
        squeeze_output = True
    elif signal.dim() == 3:
        squeeze_output = False
    else:
        raise ValueError("signal must be a 2D tensor of shape (T, C) or 3D tensor of shape (B, T, C)")
    
    B, T, C = signal.size()
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
    left = signal[:, :pad].flip(dims=(1,))           
    right = signal[:, -pad:].flip(dims=(1,))        
    f = torch.cat([left, signal, right], dim=1)   # (B, L, C)
    L = f.size(1)  # 2T if T is even else 2T - 1 
    freqs = torch.arange(L, device=device, dtype=dtype) / L - 0.5  # (L,)
    # Construct and center f_hat
    f_hat_plus = torch.fft.fftshift(torch.fft.fft(f, dim=1), dim=1)  # (B, L, C)
    f_hat_plus[:, :L // 2] = 0

    """Initialization"""
    # matrix keeping track of every iterant
    u_hat_plus = torch.zeros(B, L, C, K, dtype=ctype, device=device)  # (B, L, C, K)
    u_hat_prev = u_hat_plus.clone()  # (B, L, C, K)
    omega_plus = []
    # initialize omegas uniformly
    if init == 1:
        omega = ((0.5 / K) * torch.arange(K, dtype=dtype, device=device)).unsqueeze(0).expand(B, K).contiguous()
    elif init == 2:
        omega = ((math.log(0.5 * T) * torch.rand(B, K, dtype=dtype, device=device)).exp() / T).sort(dim=-1)[0]
    else:
        omega = torch.zeros(B, K, dtype=dtype, device=device)
    # if DC mode imposed, set its omega to 0
    if DC:
        omega[:, 0] = 0
    omega_plus.append(omega)  # (B, K)

    """Algorithm of MVMD"""
    lambda_hat = torch.zeros(B, L, C, dtype=ctype, device=device)  # (B, L, C)
    sum_uk = torch.zeros(B, L, C, dtype=ctype, device=device)  # (B, L, C)
    for _ in range(N - 1):
        # update modes
        n = f_hat_plus - 0.5 * lambda_hat  # (B, L, C)
        d = (1 + alpha * (freqs.view(1, L, 1, 1) - omega_plus[-1].view(B, 1, 1, K)).square()).reciprocal() # (1, L, 1, K)
        for k in range(K):
            # When k is 0, u_hat_plus[..., k - 1] is u_hat_prev[..., -1].
            sum_uk += u_hat_plus[..., k - 1] - u_hat_prev[..., k]
            u_hat_plus[..., k] = (n - sum_uk) * d[..., k]
        # center frequencies
        power = u_hat_plus[:, L // 2:].abs().square() # (B, L/2, C, K)
        omega = torch.einsum('l,blck->bk', freqs[L // 2:], power) / (power.sum(dim=(1, 2)) + torch.finfo(dtype).eps)
        if DC:
            omega[:, 0] = 0
        omega_plus.append(omega)
        # Dual ascent
        lambda_hat += tau * (u_hat_plus.sum(dim=-1) - f_hat_plus)
        # convergence
        u_diff = (u_hat_plus - u_hat_prev).abs().square().sum().item() / (B * L) + torch.finfo(float).eps
        u_hat_prev = u_hat_plus.clone()
        if u_diff <= tol:
            break

    """Post-processing and cleanup"""
    # discard empty space if converged early
    omega = torch.stack(omega_plus, dim=1)
    # Signal reconstruction
    u_hat = torch.empty_like(u_hat_plus)
    u_hat[:, L // 2:] = u_hat_plus[:, L // 2:]
    u_hat[:, 1:L // 2 + 1] = u_hat_plus[:, L // 2:].conj().flip(dims=(1,))
    u_hat[:, 0] = u_hat[:, -1].conj()

    u = torch.fft.ifft(torch.fft.ifftshift(u_hat, dim=1), dim=1).real
    # remove mirror part 
    u = u[:, pad:-pad]  # (B, T, C, K)
    # recompute spectrum
    u_hat = torch.fft.fftshift(torch.fft.fft(u, dim=1), dim=1)

    # Squeeze output if input was non-batched
    if squeeze_output:
        u, u_hat, omega = u.squeeze(0), u_hat.squeeze(0), omega.squeeze(0)
    
    return u, u_hat, omega


if __name__ == "__main__":
    B, T = 2, 1000
    t = torch.linspace(1 / T, 1, T, dtype=float)
    f_channel1 = torch.cos(2*torch.pi*2*t) + 1/16*torch.cos(2*torch.pi*36*t)
    f_channel2 = 1/4*torch.cos(2*torch.pi*24*t) + 1/16*torch.cos(2*torch.pi*36*t)
    signal = torch.stack([f_channel1, f_channel2], dim=1)
    u, u_hat, omega = mvmd(signal.unsqueeze(0).expand(B, *signal.shape), 3)

u
