import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import create_dataset
from models import MLPModel, RNNModel, LSTMModel


def to_one_hot(labels, num_classes=4):
    one_hot = np.zeros((len(labels), num_classes))

    for i in range(len(labels)):
        one_hot[i, labels[i] - 1] = 1

    return one_hot


def calculate_accuracy(outputs, labels):
    predictions = torch.argmax(outputs, dim=1) + 1
    correct = predictions == labels
    return correct.float().mean().item(), predictions


def train_model(model_name, model, x_train, y_train, labels):
    print("\nTraining", model_name)

    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(151):
        optimizer.zero_grad()

        outputs = model(x_train)
        loss = loss_function(outputs, y_train)

        loss.backward()
        optimizer.step()

        if epoch % 30 == 0:
            accuracy, predictions = calculate_accuracy(outputs, labels)
            print("Epoch:", epoch, "Loss:", loss.item(), "Accuracy:", accuracy)

    outputs = model(x_train)
    accuracy, predictions = calculate_accuracy(outputs, labels)

    print("Final accuracy:", accuracy)
    print("True labels:")
    print(labels[:10].numpy())

    print("Predictions:")
    print(predictions[:10].detach().numpy())


X, y = create_dataset(300)
y_one_hot = to_one_hot(y)

x_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y_one_hot, dtype=torch.float32)
labels_tensor = torch.tensor(y, dtype=torch.long)

models = [
    ("MLP", MLPModel()),
    ("RNN", RNNModel()),
    ("LSTM", LSTMModel()),
]

for name, model in models:
    train_model(name, model, x_tensor, y_tensor, labels_tensor)