import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import torch

# Setup a hypothetical optimizer with a learning rate
optimizer1 = torch.optim.Adam([torch.zeros(3, requires_grad=True)], lr=0.01)

# Setup for CosineAnnealingLR
scheduler_cosine = CosineAnnealingLR(optimizer1, T_max=50, eta_min=0)

# Setup for CosineAnnealingWarmRestarts
optimizer2 = torch.optim.Adam([torch.zeros(3, requires_grad=True)], lr=0.01)
scheduler_cosine_warm = CosineAnnealingWarmRestarts(optimizer2, T_0=50, T_mult=1, eta_min=0)

# Number of epochs for simulation
num_epochs = 50

# Recording the learning rates
lr_cosine, lr_cosine_warm = [], []
scheduler_cosine_warm.step(48)
for epoch in range(num_epochs):
    scheduler_cosine.step()
    lr_cosine.append(optimizer1.param_groups[0]["lr"])

    scheduler_cosine_warm.step()
    lr_cosine_warm.append(optimizer2.param_groups[0]["lr"])

    optimizer1.zero_grad()
    optimizer1.step()
    optimizer2.zero_grad()
    optimizer2.step()

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(lr_cosine, label="CosineAnnealingLR")
plt.plot(lr_cosine_warm, label="CosineAnnealingWarmRestarts")
plt.xlabel("Epochs")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedules: CosineAnnealingLR vs CosineAnnealingWarmRestarts")
plt.legend()
plt.grid(True)
plt.show()
