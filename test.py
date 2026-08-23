import torch
import numpy as np 

device = 'cpu'

mat1 = torch.rand([2, 2, 2])
mat2 = torch.rand([2, 2, 2])

mat3 = mat1 * mat2

print("mat1:\n", mat1)
print("*")
print("mat2:\n", mat2)
print("=")
print("mat3:\n", mat3)
