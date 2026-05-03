import math

import torch


def ensure_finite_scalar(name, value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise RuntimeError(f"{name} expected a scalar tensor, got shape {tuple(value.shape)}.")
        value = float(value.detach().cpu().item())
    else:
        value = float(value)
    if not math.isfinite(value):
        raise RuntimeError(f"{name} became non-finite: {value!r}.")
    return value


def assert_finite_tensors(name, tensors):
    for index, tensor in enumerate(tensors):
        if tensor is None:
            continue
        if not torch.isfinite(tensor.detach()).all().item():
            raise RuntimeError(f"{name}[{index}] contains non-finite values.")
    return True


def assert_finite_module(name, module):
    return assert_finite_tensors(name, (param for param in module.parameters()))


def assert_finite_gradients(name, module):
    return assert_finite_tensors(name, (param.grad for param in module.parameters() if param.grad is not None))


def optimizer_step_total(optimizer):
    total = 0
    for state in optimizer.state.values():
        step = state.get("step") if isinstance(state, dict) else None
        if step is None:
            continue
        if isinstance(step, torch.Tensor):
            total += int(step.detach().cpu().item())
        else:
            total += int(step)
    return total


def parameter_snapshot(module):
    return [param.detach().float().cpu().clone() for param in module.parameters() if param.requires_grad]


def parameter_delta_l2(module, snapshot):
    total = 0.0
    current_params = [param for param in module.parameters() if param.requires_grad]
    if len(current_params) != len(snapshot):
        raise RuntimeError("Parameter snapshot no longer matches the module parameter set.")
    for param, before in zip(current_params, snapshot):
        delta = param.detach().float().cpu() - before
        total += float(delta.pow(2).sum().item())
    return math.sqrt(total)


def ensure_optimizer_advanced(name, before_step_total, after_step_total):
    if int(after_step_total) <= int(before_step_total):
        raise RuntimeError(
            f"{name} optimizer did not advance "
            f"(before={before_step_total}, after={after_step_total})."
        )
    return True


def ensure_parameter_moved(name, delta_l2, *, min_delta=0.0):
    ensure_finite_scalar(f"{name} parameter delta", delta_l2)
    if float(delta_l2) <= float(min_delta):
        raise RuntimeError(f"{name} parameters did not change after optimizer updates.")
    return True
