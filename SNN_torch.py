"""
SNN_torch.py
- PyTorch テンソル化 Izhikevich 層と SNN (readout-only) の実装
- SurrogateHeaviside を用いた擬似勾配（オプション）
- Utilities: SimpleMLP, count_trainable_params, suggest_mlp_sizes_to_match_params

Usage:
    from SNN_torch import SNNReadoutOnly, SimpleMLP, count_trainable_params
"""
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# Default Izhikevich parameters (Regular Spiking)
DEFAULT_A = 0.2
DEFAULT_B = 2
DEFAULT_C = -56.0
DEFAULT_D = -13.0
DEFAULT_DT = 1.0
DEFAULT_V_TH = 30.0

#RS:（a=0.02, b=0.2, c=-65, d=8）
#Chaotic: (a=0.2, b=2, c=-56, d=-13)


class SurrogateHeaviside(torch.autograd.Function):
    """Piecewise-linear surrogate gradient for Heaviside step."""
    @staticmethod
    def forward(ctx, x: torch.Tensor, gamma: float = 1.0):
        ctx.save_for_backward(x)
        ctx.gamma = float(gamma)
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        gamma = ctx.gamma
        grad_input = torch.clamp(1.0 - x.abs() / gamma, min=0.0) / gamma
        return grad_output * grad_input, None


surrogate_spike = SurrogateHeaviside.apply


class IzhikevichLayerTorch(nn.Module):
    """
    Vectorized Izhikevich neuron layer (batch x units).
    - reset_state(batch_size, device, dtype) must be called before stepping.
    - step(I) takes I: (B, H) and returns spikes: (B, H) (float tensor 0/1).
    """
    def __init__(self,
                 size: int,
                 a: float = DEFAULT_A,
                 b: float = DEFAULT_B,
                 c: float = DEFAULT_C,
                 d: float = DEFAULT_D,
                 dt: float = DEFAULT_DT,
                 v_th: float = DEFAULT_V_TH,
                 use_surrogate: bool = False,
                 surrogate_gamma: float = 1.0):
        super().__init__()
        self.size = int(size)
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)
        self.dt = float(dt)
        self.v_th = float(v_th)
        self.use_surrogate = bool(use_surrogate)
        self.surrogate_gamma = float(surrogate_gamma)

        # state buffers (created in reset_state)
        # Note: buffers may be None initially
        self.register_buffer("_v", None)
        self.register_buffer("_u", None)

    def reset_state(self, batch_size: int, device: Optional[torch.device] = None, dtype=torch.float32):
        device = device or torch.device("cpu")
        v0 = torch.full((batch_size, self.size), self.c, device=device, dtype=dtype)
        u0 = (self.b * v0).to(dtype)
        # store as buffers
        self._v = v0
        self._u = u0

    @property
    def v(self):
        return self._v

    @property
    def u(self):
        return self._u

    def step(self, I: torch.Tensor) -> torch.Tensor:
        """
        Update v,u by Euler and return spike tensor.
        I: (B,H) tensor on same device/dtype
        """
        v = self._v
        u = self._u
        # compute dv, du
        dv = (0.04 * v * v + 5.0 * v + 140.0 - u + I) * self.dt
        du = (self.a * (self.b * v - u)) * self.dt
        v = v + dv
        u = u + du

        # spike detection (after update)
        if self.use_surrogate:
            s = surrogate_spike(v - self.v_th, self.surrogate_gamma)
        else:
            s = (v >= self.v_th).to(v.dtype)

        # reset where spiked
        if s.any():
            mask = s.bool()
            v = torch.where(mask, torch.full_like(v, self.c), v)
            u = torch.where(mask, u + self.d, u)

        # save state
        self._v = v
        self._u = u
        return s


class SNNReadoutOnly(nn.Module):
    """
    Readout-only SNN:
    - fixed random projection W1 (registered as buffer, not trained)
    - Izhikevich hidden layer (vectorized)
    - accumulate spikes across T steps -> optional centre/normalise -> linear readout
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 T: int = 200,
                 input_scale: float = 1.0,
                 input_bias: float = 0.0,
                 normalize_counts: bool = True,
                 center_counts: bool = True,
                 seed: Optional[int] = 42,
                 izhikevich_params: Optional[dict] = None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(int(seed))

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.T = int(T)
        self.input_scale = float(input_scale)
        self.input_bias = float(input_bias)
        self.normalize_counts = bool(normalize_counts)
        self.center_counts = bool(center_counts)

        # fixed random projection W1 (buffer, not trainable) and bias b1
        W1 = torch.randn(self.input_dim, self.hidden_dim) * (1.0 / math.sqrt(self.input_dim))
        b1 = torch.zeros(self.hidden_dim)
        self.register_buffer("W1", W1)
        self.register_buffer("b1", b1)

        ip = izhikevich_params or {}
        self.izh = IzhikevichLayerTorch(self.hidden_dim,
                                       a=ip.get("a", DEFAULT_A),
                                       b=ip.get("b", DEFAULT_B),
                                       c=ip.get("c", DEFAULT_C),
                                       d=ip.get("d", DEFAULT_D),
                                       dt=ip.get("dt", DEFAULT_DT),
                                       v_th=ip.get("v_th", DEFAULT_V_TH),
                                       use_surrogate=ip.get("use_surrogate", False),
                                       surrogate_gamma=ip.get("surrogate_gamma", 1.0))

        # readout layer (trainable)
        self.readout = nn.Linear(self.hidden_dim, self.output_dim)

    def forward_counts(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, D) tensor in [0,1]
        returns processed S_h: (B, H)
        """
        B = x.shape[0]
        device = x.device
        dtype = x.dtype
        # reset state
        self.izh.reset_state(B, device=device, dtype=dtype)
        # constant input current per time step
        I_h = x @ self.W1.to(device=device) + self.b1.to(device=device)
        if self.input_bias != 0.0:
            I_h = I_h + self.input_bias
        I_h = self.input_scale * I_h
        S_h = torch.zeros_like(I_h, device=device, dtype=dtype)
        for _ in range(self.T):
            s = self.izh.step(I_h)
            S_h = S_h + s
        # center and normalise
        if self.center_counts:
            S_h = S_h - S_h.mean(dim=0, keepdim=True)
        if self.normalize_counts:
            S_h = S_h / max(self.T, 1)
        return S_h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S_h = self.forward_counts(x)
        logits = self.readout(S_h)
        return logits


class SimpleMLP(nn.Module):
    """2-layer MLP baseline: fc1 -> relu -> dropout -> fc2 -> relu -> fc3"""
    def __init__(self, input_dim: int, hidden1: int, hidden2: int, output_dim: int, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def count_trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def suggest_mlp_sizes_to_match_params(target_trainable: int, input_dim: int = 784, output_dim: int = 10) -> Tuple[int, int]:
    """
    Simple heuristic: solve approximate quadratic for h (h1 = h2 = h)
    target ≈ input*h + h + h*h + h + h*out  => h^2 + h*(input+out+2) - target ≈ 0
    """
    a = 1.0
    b = (input_dim + output_dim + 2)
    c = -float(target_trainable)
    disc = b * b - 4 * a * c
    if disc <= 0:
        h = int(max(16, target_trainable // (input_dim + output_dim + 1)))
    else:
        h = int(max(16, (-b + math.sqrt(disc)) / 2.0))
    return h, h