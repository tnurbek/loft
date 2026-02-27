import torch
from torch.nn.utils import clip_grad_norm_

from .optimizer import LoFTAdamW
from .optim_helper import (
    get_total_norm, clip_grads_with_norm_,
    rescale_gradients, compute_full_grad_from_proj,
    do_update_)


@torch.no_grad()
def clip_grad_norm(model, optimizer, max_norm: float,
                   norm_type: float = 2.0,
                   error_if_nonfinite: bool = False) -> None:
    """
    Clips gradient norm of an iterable of parameters.
    The norm is computed over all gradients together, as opposed to
    per-parameter or per-layer.
    :param model: The model to clip gradients for.
    :param optimizer: The optimizer to clip gradients for.
    :param max_norm: The maximum norm of the gradients.
    :param norm_type: The type of norm to use. Default is 2.0.
    :param error_if_nonfinite: Whether to raise an error if the norm is non-finite.
    """
    if max_norm is None:
        return
    parameters = model.parameters()
    if isinstance(optimizer, LoFTAdamW):
        lora_A_name = optimizer.lora_A_name
        grads = []
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                # check whether the parameter is a LoRA parameter
                p_name = optimizer.lora_params_to_name[p]
                # check whether to do update
                do_update = do_update_(optimizer, p_name)
                if not do_update:
                    continue

                # perform reprojections for LoRA parameters
                if optimizer.rescale_grads and p_name is not None:
                    lora_A = lora_A_name in p_name
                    # get other parameter
                    other_p_name = optimizer._get_other_param_name(p_name)
                    other_p = optimizer.lora_name_to_params[other_p_name]
                    assert len(other_p.shape) == 2
                    # gradient rescaling
                    grad, _ = rescale_gradients(grad, other_p, lora_A)
                    # print(f"rescaling {p_name} with grad")

                if optimizer.reproject_second_moment and p_name is not None:
                    full_grad = compute_full_grad_from_proj(
                        grad, other_p, lora_A)
                    # print(f"clipping {p_name} with full_grad")
                    grads.append(full_grad)
                else:
                    grads.append(grad)

        # print grad norms
        # for i, grad in enumerate(grads):
        #     norm = torch.linalg.vector_norm(grad, norm_type)
        #     print(f"grad norm {i} {norm} ... {grad.shape}")

        total_norm = get_total_norm(
            grads, norm_type, error_if_nonfinite)
        # print(f"total norm (scaling  ) {total_norm}")
        clip_grads_with_norm_(
            parameters, max_norm, total_norm)
    else:
        total_norm = clip_grad_norm_(
            parameters, max_norm, norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite)
        # print(f"total norm (no scaling) {total_norm}")
