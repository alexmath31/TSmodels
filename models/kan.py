import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class KAN(nn.Module):
    def __init__(self, input_size):
        super(KAN, self).__init__()
        self.univariate_funcs = nn.ModuleList([nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        ) for _ in range(input_size)])

        self.combine_funcs = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        univariate_outputs = []
        for i in range(x.size(1)):
            xi = x[:, i].unsqueeze(1)
            univariate_output = self.univariate_funcs[i](xi)
            univariate_outputs.append(univariate_output)
        concat_outputs = torch.cat(univariate_outputs, dim=1)
        out = self.combine_funcs(concat_outputs)
        return out

def forecast_kan(train, steps):
    kan = KAN(num_nodes=10)
    
    X_train = np.arange(len(train)).reshape(-1, 1)
    y_train = train.values

    kan.fit(X_train, y_train, epochs=500)
    X_test = np.arange(len(train), len(train) + steps).reshape(-1, 1)
    preds = kan.predict(X_test)

    return pd.Series(preds, index=pd.date_range(start=train.index[-1], periods=steps + 1, freq="D")[1:])
