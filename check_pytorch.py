import torch
print(torch.__version__, torch.cuda.is_available())
x = torch.rand(5, 3)
print(x)
