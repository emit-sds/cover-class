import torch


def one_hot_encode_simulated_data(y:torch.Tensor, num_classes:int) -> torch.Tensor:
    out = torch.zeros(y.size(0), num_classes, dtype=torch.double, device=y.device)
    out.scatter_add_(dim=1, index=y.clamp(0), src=(y >= 0).double())
    out.clamp_(max=1)
    return out
