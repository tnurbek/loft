import functools
from typing import Dict, List, Optional, Tuple, Union, Iterable

import torch
from torch import Tensor
from torch.utils._foreach_utils import (
    _device_has_foreach_support,
    _group_tensors_by_device_and_dtype,
    _has_foreach_support,
)

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]


# functions to transform gradients and moments
def rescale_gradients(grad, other_p, lora_A):
    if lora_A:
        eye = torch.eye(other_p.size(1)).to(other_p.device)
        scaling = torch.linalg.inv(
            other_p.T @ other_p + 1e-6 * eye)
        grad = scaling @ grad
    else:  # lora_B
        eye = torch.eye(other_p.size(0)).to(other_p.device)
        scaling = torch.linalg.inv(
            other_p @ other_p.T + 1e-8 * eye)
        grad = grad @ scaling
    return grad, scaling


def get_mom_reprojection(other_p, old_other_p, scaling, lora_A):
    if lora_A:
        R = scaling @ (other_p.T @ old_other_p)
    else:  # lora_B
        R = (old_other_p @ other_p.T) @ scaling
    return R


def get_reprojected_momentum(R, exp_avg, lora_A):
    # reproject momentum
    if lora_A:
        return R @ exp_avg
    else:  # lora_B
        return exp_avg @ R


def get_reprojected_second_moment(R, row_prod, lora_A):
    # reproject second moment
    if lora_A:
        return R.T[None, :, :] @ row_prod @ R[None, :, :]
    else:  # lora_B
        return R[None, :, :] @ row_prod @ R.T[None, :, :]


# def compute_mom_sq_from_row_prod_and_transformation_old(row_prod, transformation):
#     """
#     Compute the second moment of the momentum from the row product and
#       the transformation matrix.
#     :param row_prod: The row product of the matrix.
#     :param transformation: The transformation matrix.
#     :return: The second moment of the momentum.
#     """
#     n, d, r = row_prod.size(0), transformation.size(1), row_prod.size(1)
#     print(f"Shapes: row_prod {row_prod.shape}, transformation {transformation.shape}")
#     X = torch.bmm(row_prod, transformation.unsqueeze(0).expand(
#         row_prod.size(0), -1, d))
#     Y = torch.bmm(transformation.transpose(0, 1).unsqueeze(0).expand(
#         n, d, r), X)
#     print(f"Shapes: X {X.shape}, Y {Y.shape}")
#     mom_sq = Y.diagonal(dim1=1, dim2=2)
#     # clamp all values to be positive to avoid NaNs
#     return mom_sq.clamp(min=0)

def compute_mom_sq_from_row_prod_and_transformation(row_prod, transformation):
    """
    Efficient computation of second moment from row products
      and transformation.
    :param row_prod: Tensor of shape (n, r, r)
    :param transformation: Tensor of shape (r, d)
    :return: Tensor of shape (n, d)
    """
    # compute intermediate product: (n, r, d)
    temp = torch.matmul(row_prod, transformation)  # (n, r, d)
    # sum over r of Tᵢ * (P @ T)ᵢ = element-wise product then sum
    mom_sq = (transformation.unsqueeze(0) * temp).sum(dim=1)  # (n, d)

    return mom_sq.clamp(min=0)


def compute_full_grad_from_proj(proj_grad, other_p, lora_A):
    if lora_A:
        return other_p @ proj_grad
    else:  # lora_B
        return proj_grad @ other_p


def compute_proj_update(full_update, scaling, other_p, lora_A):
    if lora_A:
        return scaling @ (other_p.T @ full_update)
    else:  # lora_B
        return (full_update @ other_p.T) @ scaling


def do_update_(optimizer, p_name):
    do_update = True
    lora_A_name = optimizer.lora_A_name
    lora_B_name = optimizer.lora_B_name
    if optimizer.alternate_update and p_name is not None:
        if optimizer.update_A and (lora_B_name in p_name):
            do_update = False
        if not optimizer.update_A and (lora_A_name in p_name):
            do_update = False
    return do_update


# from torch 2.6.0
def _no_grad(func):
    """
    This wrapper is needed to avoid a circular import when using
      @torch.no_grad on the exposed functions
    clip_grad_norm_ and clip_grad_value_ themselves.
    """

    def _no_grad_wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)

    functools.update_wrapper(_no_grad_wrapper, func)
    return _no_grad_wrapper


@_no_grad
def get_total_norm(
    tensors: _tensor_or_tensors,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
) -> torch.Tensor:
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    else:
        tensors = list(tensors)
    norm_type = float(norm_type)
    if len(tensors) == 0:
        return torch.tensor(0.0)
    first_device = tensors[0].device
    grouped_tensors: Dict[
        Tuple[torch.device, torch.dtype], Tuple[List[List[Tensor]], List[int]]
    ] = _group_tensors_by_device_and_dtype(
        [tensors]  # type: ignore[list-item]
    )  # type: ignore[assignment]

    norms: List[Tensor] = []
    for (device, _), ([device_tensors], _) in grouped_tensors.items():
        if (foreach is None and _has_foreach_support(device_tensors, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            norms.extend(torch._foreach_norm(device_tensors, norm_type))
        elif foreach:
            raise RuntimeError(
                f"foreach=True was passed, but can't use the foreach API on {device.type} tensors"
            )
        else:
            norms.extend(
                [torch.linalg.vector_norm(g, norm_type) for g in device_tensors]
            )

    total_norm = torch.linalg.vector_norm(
        torch.stack([norm.to(first_device) for norm in norms]), norm_type
    )

    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )
    return total_norm


@_no_grad
def clip_grads_with_norm_(
    parameters: _tensor_or_tensors,
    max_norm: float,
    total_norm: torch.Tensor,
    foreach: Optional[bool] = None,
) -> None:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    if len(grads) == 0:
        return
    grouped_grads: Dict[
        Tuple[torch.device, torch.dtype], Tuple[List[List[Tensor]], List[int]]
    ] = _group_tensors_by_device_and_dtype(
        [grads]
    )  # type: ignore[assignment]

    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for (device, _), ([device_grads], _) in grouped_grads.items():
        if (foreach is None and _has_foreach_support(device_grads, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            torch._foreach_mul_(device_grads, clip_coef_clamped.to(device))
        elif foreach:
            raise RuntimeError(
                f"foreach=True was passed, but can't use the foreach API on {device.type} tensors"
            )
        else:
            clip_coef_clamped_device = clip_coef_clamped.to(device)
            for g in device_grads:
                g.mul_(clip_coef_clamped_device)
