### --------------------------------------------------------------------
### prediction function (may be placed somewhere else for less overhead)
### --------------------------------------------------------------------

import torch

def predict(model, data):
    model.eval()
    parameter = next(model.parameters())
    inputs = torch.as_tensor(data, dtype=parameter.dtype, device=parameter.device)
    with torch.no_grad():
        return model(inputs).cpu().numpy()
