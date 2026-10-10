import torch

class CustomSGD(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.01, momentum: float = 0.0):
        defaults = dict(lr = lr, momentum = momentum)
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        """
        Returns the closure loss when provided, otherwise None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
    
                grad = p.grad
                state = self.state[p]
                if len(state)==0:
                    state["mom_buffer"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                buf = state["mom_buffer"]
                buf.mul_(momentum).add_(grad)
                p.add_(buf, alpha = -lr)
            
        
        return loss
