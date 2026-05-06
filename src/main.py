import torch

from dataset import create_dataset
from models import MLPModel


X, y = create_dataset(5)

x_tensor = torch.tensor(X, dtype=torch.float32)

model = MLPModel()
output = model(x_tensor)

print("Model output:")
print(output.detach().numpy())

print("Labels:")
print(y)