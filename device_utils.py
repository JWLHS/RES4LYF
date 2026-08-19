"""Device-agnostic helpers so RES4LYF runs on CUDA, Intel XPU (Arc), and CPU.

RES4LYF historically hardcoded "cuda" for device placement, RNG seeding, and
memory reporting. ComfyUI already knows the correct torch device for the
current backend (xpu/cuda/mps/cpu), so all device selection here delegates to
``comfy.model_management`` when possible and falls back to direct detection.
"""

import gc

import torch


def get_torch_device():
    """Return ComfyUI's active compute device (xpu/cuda/mps/cpu)."""
    try:
        from comfy import model_management
        return model_management.get_torch_device()
    except Exception:
        pass
    if torch.xpu.is_available():
        return torch.device("xpu", torch.xpu.current_device())
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def get_device_type(device=None):
    """Return the backend type string ('xpu', 'cuda', 'cpu', ...)."""
    return (device if device is not None else get_torch_device()).type


def is_gpu_available():
    """True when a non-CPU compute device is in use."""
    return get_torch_device().type != "cpu"


def to_device(tensor, device=None):
    """Move a tensor to the active compute device (or an explicit one)."""
    if device is None:
        device = get_torch_device()
    return tensor.to(device)


def manual_seed(seed):
    """Seed the CPU and active backend RNG (xpu/cuda)."""
    torch.manual_seed(seed)
    device_type = get_torch_device().type
    if device_type == "xpu":
        torch.xpu.manual_seed(seed)
    elif device_type == "cuda":
        torch.cuda.manual_seed(seed)


def manual_seed_all(seed):
    """Seed the CPU and all active backend RNG streams (xpu/cuda)."""
    torch.manual_seed(seed)
    device_type = get_torch_device().type
    if device_type == "xpu":
        torch.xpu.manual_seed_all(seed)
    elif device_type == "cuda":
        torch.cuda.manual_seed_all(seed)


def empty_cache():
    """Release cached memory on the active backend (xpu/cuda) or run GC on CPU."""
    device_type = get_torch_device().type
    if device_type == "xpu":
        torch.xpu.empty_cache()
    elif device_type == "cuda":
        torch.cuda.empty_cache()
    else:
        gc.collect()


def _backend_module(device):
    device_type = get_device_type(device)
    if device_type == "xpu":
        return torch.xpu, "xpu"
    return torch.cuda, "cuda"


def reset_peak_memory_stats(device=None):
    backend, _ = _backend_module(device)
    backend.reset_peak_memory_stats(device)


def max_memory_allocated(device=None):
    backend, _ = _backend_module(device)
    return backend.max_memory_allocated(device)


def max_memory_reserved(device=None):
    backend, _ = _backend_module(device)
    return backend.max_memory_reserved(device)


def get_total_memory(device=None):
    backend, _ = _backend_module(device)
    return backend.get_device_properties(device).total_memory


def set_memory_fraction(fraction, device=None):
    backend, _ = _backend_module(device)
    backend.set_per_process_memory_fraction(fraction, device)


def pinv(tensor, **kwargs):
    """torch.linalg.pinv with a CPU fallback on XPU (fp64 is unsupported there)."""
    if tensor.device.type == "xpu":
        # XPU cannot run pinv in fp64 (or fp32): compute on CPU, keep fp32 on device
        return torch.linalg.pinv(tensor.cpu().float(), **kwargs).to(tensor.device)
    return torch.linalg.pinv(tensor, **kwargs)


def eigh(tensor, **kwargs):
    """torch.linalg.eigh with a CPU fallback on XPU (fp64 is unsupported there)."""
    if tensor.device.type == "xpu":
        S, U = torch.linalg.eigh(tensor.cpu(), **kwargs)
        return S.to(tensor.device), U.to(tensor.device)
    return torch.linalg.eigh(tensor, **kwargs)


def svd(tensor, **kwargs):
    """torch.linalg.svd with a CPU fallback on XPU (fp64 is unsupported there)."""
    if tensor.device.type == "xpu":
        U, S, Vh = torch.linalg.svd(tensor.cpu(), **kwargs)
        return U.to(tensor.device), S.to(tensor.device), Vh.to(tensor.device)
    return torch.linalg.svd(tensor, **kwargs)


def whitening_eigh(f_centered, eps=1e-5, use_svd=False):
    """Covariance + eigen decomposition used by the WCT feature-matching path.

    Returns (S_eig, U_eig) on the input's device. On XPU the fp64 covariance
    math is executed on CPU (XPU has no fp64 support) and the result is cast
    back to the input's dtype/device.
    """
    if f_centered.device.type == "xpu":
        work = f_centered.cpu()
        cov = (work.T.double() @ work.double()) / (work.size(0) - 1)
        cov = cov + eps * torch.eye(cov.size(0), dtype=cov.dtype, device=cov.device)
        if use_svd:
            U_svd, S_svd, _ = torch.linalg.svd(cov)
            S_eig, U_eig = S_svd, U_svd
        else:
            S_eig, U_eig = torch.linalg.eigh(cov)
        return S_eig.to(f_centered), U_eig.to(f_centered)
    cov = (f_centered.T.double() @ f_centered.double()) / (f_centered.size(0) - 1)
    cov = cov + eps * torch.eye(cov.size(0), dtype=cov.dtype, device=cov.device)
    if use_svd:
        U_svd, S_svd, _ = torch.linalg.svd(cov)
        return S_svd, U_svd
    return torch.linalg.eigh(cov)
