import torch

def gradient_accumulation(w_init: torch.Tensor, micro_batches: list, lr: float, accum_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (final_weights, last_mean_gradient) as float32 tensors.
    """
    w = w_init.clone().to(torch.float32)
    last_grad = torch.zeros_like(w)

    for start in range(0, len(micro_batches), accum_steps):
        group = micro_batches[start:start + accum_steps]
        grads = []
        for bx, by in group:
            x = torch.as_tensor(bx, dtype=torch.float32)
            y = torch.as_tensor(by, dtype=torch.float32)
            if x.dim() == 1:                       # single sample
                g = 2 * x * (w @ x - y)
            else:                                  # batch (n, d): MSE gradient
                g = 2 * x.T @ (x @ w - y) / x.shape[0]
            grads.append(g)
        last_grad = torch.stack(grads).mean(dim=0)
        w = w - lr * last_grad

    return w, last_grad
