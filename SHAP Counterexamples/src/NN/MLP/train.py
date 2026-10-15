### ---------------------------------------
### Model training for future usage (torch)
### ---------------------------------------

from pathlib import Path
import numpy as np
import torch
from model import MLP

### Load Data
caches = Path(__file__).resolve().parents[3] / "caches"
with np.load(caches / "data_cache" / "bit_data.npz") as data:
    X = torch.tensor(data["X"], dtype=torch.float32)
    y = torch.tensor(data["scores"], dtype=torch.float32)

### Initialize Model and Optimizer
model = MLP(X.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

### Train Model
for _ in range(2000):
    optimizer.zero_grad()
    loss = ((model(X) - y) ** 2).mean()
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    print("Final MSE:", ((model(X) - y) ** 2).mean().item())

model_cache = caches / "model_cache"
model_cache.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), model_cache / "model.pt")
