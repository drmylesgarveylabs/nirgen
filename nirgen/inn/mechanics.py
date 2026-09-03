"""Differentiable mechanics used by the discrete-ion INN."""

import torch


def ste_floor(value):
    """Floor in the forward pass while preserving the identity gradient."""
    return value + (torch.floor(value) - value).detach()


def ste_clamp(value, lower, upper):
    """Clamp in the forward pass while preserving the identity gradient."""
    return value + (torch.clamp(value, lower, upper) - value).detach()

