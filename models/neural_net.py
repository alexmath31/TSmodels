import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def forecast_nn(train, steps):
    X_train = torch.tensor(np.arange(len(train)).reshape(-1, 1), dtype=torch.float32)
    y_train = torch.tensor(train.values, dtype=torch.float32).view(-1, 1)

    model = SimpleNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for _ in range(500):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

    X_test = torch.tensor(np.arange(len(train), len(train) + steps).reshape(-1, 1), dtype=torch.float32)
    preds = model(X_test).detach().numpy().flatten()
    
    return pd.Series(preds, index=pd.date_range(start=train.index[-1], periods=steps + 1, freq="D")[1:])
