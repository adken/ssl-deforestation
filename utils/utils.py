"""
Optimiser and learning-rate schedule utilities for VICReg pretraining.

LARS optimiser adapted from:
https://github.com/facebookresearch/barlowtwins
"""

import math

import torch


# ── LARS optimiser ───────────────────────────────────────────────────────────

class LARS(torch.optim.Optimizer):
    """
    Layer-wise Adaptive Rate Scaling optimiser.
    Recommended for large-batch SSL pretraining (batch ≥ 256).
    """

    def __init__(
        self,
        params,
        lr: float,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        eta: float = 0.001,
        weight_decay_filter: bool = False,
        lars_adaptation_filter: bool = False,
    ) -> None:
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    def _is_bias_or_norm(self, p: torch.Tensor) -> bool:
        return p.ndim == 1

    @torch.no_grad()
    def step(self) -> None:
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad
                if dp is None:
                    continue

                if not g["weight_decay_filter"] or not self._is_bias_or_norm(p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if not g["lars_adaptation_filter"] or not self._is_bias_or_norm(p):
                    param_norm  = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0.0,
                            g["eta"] * param_norm / update_norm,
                            one,
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                state = self.state[p]
                if "mu" not in state:
                    state["mu"] = torch.zeros_like(p)
                mu = state["mu"]
                mu.mul_(g["momentum"]).add_(dp)
                p.add_(mu, alpha=-g["lr"])


# ── Learning-rate schedule ───────────────────────────────────────────────────

def adjust_learning_rate(args, optimizer: torch.optim.Optimizer, loader, step: int) -> None:
    """
    Linear warm-up for the first 10 epochs, then cosine decay to 0.1 % of peak LR.
    Peak LR is scaled linearly with batch size relative to a reference of 256.
    """
    max_steps    = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr      = args.batch_size / 256.0

    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        t = step - warmup_steps
        T = max_steps - warmup_steps
        q      = 0.5 * (1 + math.cos(math.pi * t / T))
        end_lr = base_lr * 0.001
        lr     = base_lr * q + end_lr * (1 - q)

    optimizer.param_groups[0]["lr"] = lr * args.lr


# ── Optimiser factory ────────────────────────────────────────────────────────

def optim(model: torch.nn.Module, weight_decay: float) -> LARS:
    """
    Build a LARS optimiser with separate parameter groups for weights vs
    biases/norms (the latter skip weight decay and LARS adaptation).
    """
    param_weights: list = []
    param_biases:  list = []
    for p in model.parameters():
        (param_biases if p.ndim == 1 else param_weights).append(p)

    return LARS(
        [{"params": param_weights}, {"params": param_biases}],
        lr=0,
        weight_decay=weight_decay,
        weight_decay_filter=True,
        lars_adaptation_filter=True,
    )
