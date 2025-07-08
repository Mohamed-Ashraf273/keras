import openvino as ov
import torch

x = torch.tensor([10, 20, 30])
print(x[1])  # Output: tensor(20)
print(x[0:2])  # Output: tensor([10, 20])
t = ov.Tensor([1, 2, 3])
print(t[1])
