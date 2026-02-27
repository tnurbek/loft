from transformers import TrainerCallback
import torch

from .optim_helper import clip_grad_norm

__all__ = ['GradientClippingCallback']


class GradientClippingCallback(TrainerCallback):
    def __init__(self, max_norm: float, norm_type: float = 2.0,
                 error_if_nonfinite: bool = False):
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.error_if_nonfinite = error_if_nonfinite

    @torch.no_grad()
    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        """
        Event called before the optimizer step but after gradient clipping.
        """
        optimizer = kwargs['optimizer']
        model = kwargs['model']
        max_norm = self.max_norm
        norm_type = self.norm_type
        error_if_nonfinite = self.error_if_nonfinite
        clip_grad_norm(
            model, optimizer, max_norm,
            norm_type, error_if_nonfinite)
