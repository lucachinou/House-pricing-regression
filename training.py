import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd

import arch

csv = pd.read_csv('data.csv')

size = csv['size'].tolist()
price = csv['price'].tolist()

torch.manual_seed(42)

perm = torch.randperm(len(size))
split = int(len(size) * 0.8)

x = torch.Tensor([[float(size[i])] for i in perm[:split]])
y = torch.Tensor([[float(price[i])] for i in perm[:split]])

x_eval = torch.Tensor([[float(size[i])] for i in perm[split:]])
y_eval = torch.Tensor([[float(price[i])] for i in perm[split:]])


x_norm = (x - x.mean()) / x.std()
y_norm = (y - y.mean()) / y.std()

x_eval_norm = (x_eval - x_eval.mean()) / x.std()
y_eval_norm = (y_eval - y_eval.mean()) / y.std()

print(x.mean())
print(x.std())

print(y.mean())
print(y.std())

model = arch.model()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("# Training model\n\n")
model.train()
for epoch in range(400):
    optimizer.zero_grad()

    output = model(x_norm)

    loss = criterion(output, y_norm)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss {loss.item():.4f}")

torch.save(model.state_dict(), 'model.pth')

model.eval()
print("\n\n# Evaluating model\n\n")
with torch.no_grad():
    output = model(x_eval_norm)
    loss = criterion(output, y_eval_norm)
print(f"Loss {loss.item():.4f}")