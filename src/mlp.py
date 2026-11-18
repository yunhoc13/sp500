"""A small PyTorch MLP with a scikit-learn style fit/predict interface.

It plugs into the shared pipeline exactly like Ridge or Random Forest:
scoring.score_model() calls model.fit(X_train, y_train) and model.predict(X_test)
on the already-preprocessed arrays (scaled features + sector one-hots, and the
year-demeaned target), so every model sees the same data and folds.

Overfitting controls (all inside the TRAINING rows of each split, so the test
fold is never used to decide anything):
  - early stopping on a random 15% validation slice of the training rows
  - dropout between layers
  - weight decay (L2 penalty, like Ridge)
"""
import numpy as np
import torch
from torch import nn

SEED = 42
torch.set_num_threads(2)


class TorchMLP:
    def __init__(self, hidden=(64, 32), dropout=0.1, weight_decay=1e-4, lr=1e-3,
                 max_epochs=300, batch_size=128, patience=20, val_frac=0.15, seed=SEED):
        self.hidden = tuple(hidden)
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_frac = val_frac
        self.seed = seed

    def _build(self, n_in):
        layers, prev = [], n_in
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        # hold out part of the training rows to decide when to stop
        idx = rng.permutation(len(X))
        n_val = max(1, int(len(X) * self.val_frac))
        val, tr = idx[:n_val], idx[n_val:]
        Xtr, ytr = torch.from_numpy(X[tr]), torch.from_numpy(y[tr])
        Xva, yva = torch.from_numpy(X[val]), torch.from_numpy(y[val])

        self.net = self._build(X.shape[1])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        best, best_state, waited = np.inf, None, 0
        for epoch in range(self.max_epochs):
            self.net.train()
            order = torch.from_numpy(rng.permutation(len(Xtr)))
            for start in range(0, len(Xtr), self.batch_size):
                b = order[start:start + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.net(Xtr[b]), ytr[b])
                loss.backward()
                opt.step()

            self.net.eval()
            with torch.no_grad():
                v = loss_fn(self.net(Xva), yva).item()
            if v < best - 1e-5:
                best, waited = v, 0
                best_state = {k: t.clone() for k, t in self.net.state_dict().items()}
            else:
                waited += 1
                if waited >= self.patience:
                    break

        self.net.load_state_dict(best_state)
        self.epochs_ = epoch + 1
        return self

    def predict(self, X):
        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.from_numpy(np.asarray(X, dtype=np.float32)))
        return out.numpy().ravel()
