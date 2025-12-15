import torch
import torch.nn as nn

class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature

    def calibrate(self, valid_loader, device="cpu"):
        self.model.eval()
        logits_list, labels_list = [], []

        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                logits_list.append(self.model(x))
                labels_list.append(y)

        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def closure():
            optimizer.zero_grad()
            loss = nll_criterion(logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        print(f"✅ Calibrated temperature: {self.temperature.item():.3f}")
