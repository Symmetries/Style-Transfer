import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Define neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 8)
        self.pool = nn.MaxPool2d(8, 8)
        self.conv2 = nn.Conv2d(6, 16, 8)
        self.fc1 = nn.Linear(16 * 7 * 7, 520)
        self.fc2 = nn.Linear(520, 190)
        self.fc3 = nn.Linear(190, 90)

    def forward(self, x):
        print(x.size())
        print(self.conv1(x).size())
        x = self.pool(F.relu(self.conv1(x)))
        print(x.size())
        print(self.conv2(x).size())
        x = self.pool(F.relu(self.conv2(x)))
        print(x.size())
        x = x.view(-1, 16 * 7 * 7)
        print(x.size())
        x = F.relu(self.fc1(x))
        print(x.size())
        x = F.relu(self.fc2(x))
        print(x.size())
        x = torch.sigmoid(self.fc3(x))
        print(x.size())
        return x

net = Net()

# Use Binary Cross Entropy (since it's multilabel classification)
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9)

# Load the data loader
valloader = torch.load('valloader.pt')

# Test overfitting on a single batch
for images, labels in valloader:
    for _ in range(30):
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            outputs = outputs > 0.5
            r = (outputs == labels.byte())
            acc = r.float().sum().item()
            acc = float(acc) / (32 * 90)
            print('Loss: {}, Acc: {}'.format(float(loss), float(acc)))
    break
