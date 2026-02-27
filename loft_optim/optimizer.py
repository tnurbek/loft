import math
import torch
from torch.optim import Optimizer

from collections import defaultdict

from .optim_helper import (rescale_gradients, get_mom_reprojection,
                           get_reprojected_momentum,
                           get_reprojected_second_moment,
                           compute_mom_sq_from_row_prod_and_transformation,
                           compute_full_grad_from_proj, compute_proj_update,
                           do_update_)

__all__ = ['AdamW', 'LoFTAdamW']


# define AdamW optimizer
class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'AdamW does not support sparse gradients')

                amsgrad = group['amsgrad']
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']

                beta1, beta2 = group['betas']

                state['step'] += 1
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if amsgrad:
                    torch.maximum(max_exp_avg_sq, exp_avg_sq,
                                  out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(
                        bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(
                        bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'])

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class LoFTAdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-4,
                 weight_decay=1e-2, amsgrad=False,
                 lora_A_name="lora_A", lora_B_name="lora_B",
                 alternate_update: bool = True,
                 rescale_grads: bool = True,
                 reproject_momentum: bool = True,
                 reproject_second_moment: bool = True,
                 model: torch.nn.Module = None):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)
        if amsgrad:
            raise ValueError("LoRAFTAdamW does not support amsgrad.")
        self.lora_A_name = lora_A_name
        self.lora_B_name = lora_B_name
        self.alternate_update = alternate_update
        # update B first as it is usually initialized to zero
        self.update_A = False
        self.rescale_grads = rescale_grads
        self.reproject_momentum = reproject_momentum
        self.reproject_second_moment = reproject_second_moment
        lora_params_to_name, lora_name_to_params = \
            self._lora_params_to_name(model)
        self.lora_params_to_name = lora_params_to_name
        self.lora_name_to_params = lora_name_to_params
        self.old_params = dict()

    def _lora_params_to_name(self, model):
        lora_params_to_name = defaultdict(lambda: None)
        lora_name_to_params = defaultdict(lambda: None)
        for name, param in model.named_parameters():
            if param.requires_grad:
                if self.lora_A_name in name or self.lora_B_name in name:
                    if "weight" in name:
                        lora_params_to_name[param] = name
                        lora_name_to_params[name] = param
        return lora_params_to_name, lora_name_to_params

    def _get_other_param_name(self, param_name):
        lora_A_name = self.lora_A_name
        lora_B_name = self.lora_B_name

        if lora_A_name in param_name:
            return param_name.replace(lora_A_name, lora_B_name)
        elif "lora_B" in param_name:
            return param_name.replace(lora_B_name, lora_A_name)
        else:
            raise ValueError(f"Invalid Lora parameter name: {param_name}")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        lora_A_name = self.lora_A_name

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'LoRAFTAdamW does not support sparse gradients')
                state = self.state[p]

                # check whether the parameter is a LoRA parameter
                p_name = self.lora_params_to_name[p]
                # check whether to do update
                do_update = do_update_(self, p_name)

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    if self.reproject_second_moment and p_name is not None:
                        if lora_A_name in p_name:
                            r, n = p.shape
                            state["row_products"] = torch.zeros(
                                n, r, r).to(p.device)
                        else:  # lora_B
                            m, r = p.shape
                            state["row_products"] = torch.zeros(
                                m, r, r).to(p.device)
                    else:
                        state['exp_avg_sq'] = torch.zeros_like(p.data)

                # perform reprojections for LoRA parameters
                if self.rescale_grads and p_name is not None:
                    lora_A = lora_A_name in p_name
                    # get other parameter
                    other_p_name = self._get_other_param_name(p_name)
                    other_p = self.lora_name_to_params[other_p_name]
                    assert len(other_p.shape) == 2
                    # gradient rescaling
                    grad, scaling = rescale_gradients(grad, other_p, lora_A)
                    # reproject momentum and second moment
                    if self.reproject_momentum:
                        exp_avg = state['exp_avg']
                        if self.old_params.get(other_p_name, None) is not None:
                            old_other_p = self.old_params[other_p_name]
                            # construct reprojection matrix
                            R = get_mom_reprojection(
                                other_p, old_other_p, scaling, lora_A)
                            # reproject momentum
                            state['exp_avg'] = get_reprojected_momentum(
                                R, exp_avg, lora_A)
                            # reproject second moment
                            if self.reproject_second_moment:
                                row_prod = state["row_products"]
                                state["row_products"] = \
                                    get_reprojected_second_moment(
                                        R, row_prod, lora_A)
                        # store parameters used for reprojection
                        self.old_params[other_p_name] = \
                            other_p.data.detach().clone()

                beta1, beta2 = group['betas']
                state['step'] += 1
                exp_avg = state['exp_avg']
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                if self.reproject_second_moment and p_name is not None:
                    row_prod = state["row_products"]
                    vec = grad.T if lora_A else grad
                    row_prod.mul_(beta2).baddbmm_(
                        vec[:, :, None], vec[:, None, :], alpha=1 - beta2)
                else:
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg_sq.mul_(beta2).addcmul_(
                        grad, grad, value=1 - beta2)

                # skip update if doing alternate update and not updating
                if not do_update:
                    # print(f"Skipping update for {p_name}")
                    continue

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # weight decay
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'])

                step_size = group['lr'] / bias_correction1

                if self.reproject_second_moment and p_name is not None:
                    full_update = compute_full_grad_from_proj(
                        exp_avg, other_p, lora_A)
                    denom = (compute_mom_sq_from_row_prod_and_transformation(
                        row_prod, other_p.T if lora_A else other_p).sqrt() /
                        math.sqrt(bias_correction2)).add_(group['eps'])
                    full_update.div_(denom.T if lora_A else denom)
                    proj_update = compute_proj_update(
                        full_update, scaling, other_p, lora_A)
                    p.data.add_(proj_update, alpha=-step_size)
                    pass
                else:
                    denom = (exp_avg_sq.sqrt() /
                             math.sqrt(bias_correction2)).add_(group['eps'])
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

        # Alternate between updating A and B
        if self.alternate_update:
            self.update_A = not self.update_A

        return loss
