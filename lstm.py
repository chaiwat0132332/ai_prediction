# ==========================================================
# Production-grade LSTM Model for Time Series Forecasting
# ==========================================================

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler


# ==========================================================
# Network Architecture
# ==========================================================

class LSTMNetwork(nn.Module):

    def __init__(
        self,
        input_size=1,
        hidden_size=128,
        num_layers=2,
        dropout=0.2
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

        self.init_weights()

    # ======================================================
    # Proper weight initialization
    # ======================================================

    def init_weights(self):

        for name, param in self.lstm.named_parameters():

            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

            elif "weight_hh" in name:
                nn.init.orthogonal_(param)

            elif "bias" in name:
                nn.init.zeros_(param)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    # ======================================================
    # Forward
    # ======================================================

    def forward(self, x):

        out, _ = self.lstm(x)

        out = out[:, -1, :]

        out = self.fc(out)

        return out


# ==========================================================
# Model Wrapper
# ==========================================================

class LSTMModel:

    def __init__(
        self,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        lr=0.001,
        batch_size=32,
        epochs=100,
        patience=15,
        device=None
    ):

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = LSTMNetwork(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)

        self.scaler = MinMaxScaler()

        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr
        )

        self.criterion = nn.MSELoss()

        self.learning_curve = {
            "train_loss": [],
            "val_loss": []
        }

        self.is_fitted = False

    # ======================================================
    # Window Creation (CRITICAL)
    # ======================================================

    def create_windows(self, series, lag):

        X = []
        y = []

        for i in range(len(series) - lag):

            X.append(series[i:i+lag])
            y.append(series[i+lag])

        return np.array(X), np.array(y)

    # ======================================================
    # Fit
    # ======================================================

    def fit(self, series, lag, val_ratio=0.2):

        series = np.array(series)

        # Normalize entire series safely
        series_scaled = self.scaler.fit_transform(
            series.reshape(-1, 1)
        ).flatten()

        # Create windows
        X, y = self.create_windows(series_scaled, lag)

        split = int(len(X) * (1 - val_ratio))

        X_train = X[:split]
        y_train = y[:split]

        X_val = X[split:]
        y_val = y[split:]

        X_train = torch.FloatTensor(X_train).unsqueeze(-1)
        y_train = torch.FloatTensor(y_train)

        X_val = torch.FloatTensor(X_val).unsqueeze(-1)
        y_val = torch.FloatTensor(y_val)

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.batch_size,
            shuffle=False
        )

        val_loader = DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=self.batch_size,
            shuffle=False
        )

        best_loss = float("inf")
        patience_counter = 0

        # ==================================================
        # Training Loop
        # ==================================================

        for epoch in range(self.epochs):

            self.model.train()

            train_loss = 0

            for xb, yb in train_loader:

                xb = xb.to(self.device)
                yb = yb.to(self.device)

                self.optimizer.zero_grad()

                pred = self.model(xb).squeeze()

                loss = self.criterion(pred, yb)

                loss.backward()

                # Gradient clipping (CRITICAL)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0
                )

                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            self.model.eval()

            val_loss = 0

            with torch.no_grad():

                for xb, yb in val_loader:

                    xb = xb.to(self.device)
                    yb = yb.to(self.device)

                    pred = self.model(xb).squeeze()

                    loss = self.criterion(pred, yb)

                    val_loss += loss.item()

            val_loss /= len(val_loader)

            self.learning_curve["train_loss"].append(train_loss)
            self.learning_curve["val_loss"].append(val_loss)

            # Early stopping
            if val_loss < best_loss:

                best_loss = val_loss
                patience_counter = 0

                self.best_state = {
                    k: v.cpu()
                    for k, v in self.model.state_dict().items()
                }

            else:

                patience_counter += 1

                if patience_counter >= self.patience:
                    break

        # restore best weights
        self.model.load_state_dict(self.best_state)

        self.is_fitted = True

        return self.learning_curve

    # ======================================================
    # Predict
    # ======================================================

    def predict(self, X):

        X_scaled = self.scaler.transform(
            X.reshape(-1, 1)
        ).reshape(X.shape)

        X_tensor = torch.FloatTensor(X_scaled)\
            .unsqueeze(-1)\
            .to(self.device)

        self.model.eval()

        with torch.no_grad():

            pred = self.model(X_tensor)\
                .cpu().numpy()

        pred = self.scaler.inverse_transform(pred)

        return pred.flatten()

    # ======================================================
    # Forecast (recursive)
    # ======================================================

    def forecast(self, last_window, steps):

        window_scaled = self.scaler.transform(
            last_window.reshape(-1, 1)
        ).flatten()

        preds_scaled = []

        self.model.eval()

        for _ in range(steps):

            x = torch.FloatTensor(window_scaled)\
                .unsqueeze(0)\
                .unsqueeze(-1)\
                .to(self.device)

            with torch.no_grad():

                pred = self.model(x).item()

            preds_scaled.append(pred)

            window_scaled = np.roll(window_scaled, -1)
            window_scaled[-1] = pred

        preds = self.scaler.inverse_transform(
            np.array(preds_scaled).reshape(-1, 1)
        ).flatten()

        return preds

    # ======================================================
    # Learning curve getter
    # ======================================================

    def get_learning_curve(self):

        return self.learning_curve