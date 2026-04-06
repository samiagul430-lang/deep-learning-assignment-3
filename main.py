import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# -------- DATASET --------
class ColoredMNISTDataset(Dataset):
    def __init__(self, path):
        data = torch.load(path, map_location="cpu")
        self.images = data[0]
        self.labels = data[1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx].float(), self.labels[idx].long()

# -------- LOAD DATA --------
train_loader = DataLoader(ColoredMNISTDataset("train_biased.pt"), batch_size=64, shuffle=True)
test_biased_loader = DataLoader(ColoredMNISTDataset("test_biased.pt"), batch_size=64)
test_unbiased_loader = DataLoader(ColoredMNISTDataset("test_unbiased.pt"), batch_size=64)

# -------- MODEL --------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*3*3, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -------- TRAIN --------
epochs = 17
train_losses = []
train_accs = []

def accuracy(loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred==y).sum().item()
            total += y.size(0)
    return correct/total

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for x,y in train_loader:
        x,y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out,y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    acc = accuracy(train_loader)

    train_losses.append(total_loss)
    train_accs.append(acc)

    print(f"Epoch {epoch+1}, Loss={total_loss:.4f}, Acc={acc:.4f}")

# -------- TEST --------
biased_acc = accuracy(test_biased_loader)
unbiased_acc = accuracy(test_unbiased_loader)

print("Biased:", biased_acc)
print("Unbiased:", unbiased_acc)

# -------- PLOTS --------
plt.plot(train_losses)
plt.title("Training Loss")
plt.savefig("loss.png")
plt.show()

plt.plot(train_accs)
plt.title("Training Accuracy")
plt.savefig("accuracy.png")
plt.show()