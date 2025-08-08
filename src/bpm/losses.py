from __future__ import annotations
import torch
import torch.nn.functional as F


def invariance_kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """KL(p||q) between softmaxed logits p and q per-batch.
    Shapes: (B, K) -> scalar
    """
    p = F.log_softmax(p_logits, dim=-1)
    q = F.log_softmax(q_logits, dim=-1)
    p_prob = p.exp()
    return F.kl_div(q, p_prob, reduction='batchmean', log_target=False)


def embedding_l2(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(z_a, z_b)


def uniformity_kl(proto_logits: torch.Tensor) -> torch.Tensor:
    B, K = proto_logits.shape
    log_q = F.log_softmax(proto_logits, dim=-1)
    # KL(u || q) = sum u * (log u - log q) = -1/K * sum log q + const
    return -(1.0 / K) * log_q.sum(dim=-1).mean()


def byol_loss(z_student: torch.Tensor, z_teacher: torch.Tensor) -> torch.Tensor:
    z_student = F.normalize(z_student, dim=-1)
    z_teacher = F.normalize(z_teacher.detach(), dim=-1)
    return 1.0 - (z_student * z_teacher).sum(dim=-1).mean()


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_variance_covariance(z_a: torch.Tensor, z_b: torch.Tensor,
                               var_target: float = 1.0, var_weight: float = 1.0,
                               cov_weight: float = 1.0) -> torch.Tensor:
    # concatenate and center
    z = torch.cat([z_a, z_b], dim=0)
    z = z - z.mean(dim=0, keepdim=True)
    # variance penalty (lower bound)
    std = z.std(dim=0) + 1e-4
    var_loss = torch.mean(F.relu(var_target - std))
    # covariance decorrelation
    N = z.size(0)
    cov = (z.T @ z) / (N - 1)
    cov_loss = _off_diagonal(cov).pow(2).mean()
    return var_weight * var_loss + cov_weight * cov_loss
