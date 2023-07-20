
def tensor_to_cuda(tensor, device):
    if isinstance(tensor, dict):
        for key in tensor:
            tensor[key] = tensor_to_cuda(tensor[key], device)
        return tensor
    elif isinstance(tensor, (list, tuple)):
        tensor = [tensor_to_cuda(t, device) for t in tensor]
        return tensor
    else:
        return tensor.to(device)