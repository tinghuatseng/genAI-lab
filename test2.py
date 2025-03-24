import torch

print(torch.mps.is_available())

a = torch.tensor([
    [1., -1.],
    [1., -1.]
])

a.to('mps')
print(a.device)