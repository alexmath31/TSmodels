import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

def forecast_lstm(train, steps):
    train_tensor = torch.tensor(train.values, dtype=torch.float32).view(-1, 1)
    train_tensor = train_tensor.unsqueeze(0)

    model = LSTMModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for _ in range(100):
        optimizer.zero_grad()
        output = model(train_tensor)
        loss = criterion(output, train_tensor[:, -1, :])
        loss.backward()
        optimizer.step()

    preds = []
    last_seq = train_tensor[:, -1, :].detach()

    for _ in range(steps):
        with torch.no_grad():
            pred = model(last_seq.unsqueeze(0)).squeeze().item()
            preds.append(pred)
            last_seq = torch.cat((last_seq[1:], torch.tensor([[pred]])), dim=0)

    return pd.Series(preds, index=pd.date_range(start=train.index[-1], periods=steps + 1, freq="D")[1:])
