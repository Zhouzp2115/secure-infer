import torchvision
import struct
import torch
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as F


class NetworkC(nn.Module):
    def __init__(self):
        super(NetworkC, self).__init__()
        self.conv = nn.Sequential(
        nn.Conv2d(1, 16, 5,bias=False),
        nn.ReLU(True),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(16, 16, 5,bias=False),
        nn.ReLU(True),
        nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 100 ,bias=False),
            nn.ReLU(True),
            nn.Linear(100, 10,bias=False))

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def train(model, train_loader, device, batch_size):
    model.train()
    cost = torch.nn.CrossEntropyLoss()
    for epoch in range(20):
        print('train epoch ', epoch)
        for batch_idx, (data, target) in enumerate(train_loader):
           if((batch_idx + 1) * batch_size > 60000):
              break
           data, target = data.to(device), target.to(device)
           optimizer.zero_grad()
           output = model(data)
           loss = cost(output, target)
           loss.backward()
           optimizer.step()


def test(model, test_loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = F.softmax(model(data), 1)
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('\nTest set: Accuracy: {}/{} ({:.3f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def save(model):
    model = model.cpu()
    file = open('../bin/DATA/NetworkC-MNIST/weight.bin', 'wb')

    weight = model.conv[0].weight.detach()
    print('conv1')
    print(weight.max())
    print(weight.min())
    weight = weight.reshape(-1).numpy().tolist()
    for i in range(len(weight)):
       bytes = struct.pack('f', weight[i])
       file.write(bytes)

    weight = model.conv[3].weight.detach()
    print('conv2')
    print(weight.max())
    print(weight.min())
    weight = weight.reshape(-1).numpy().tolist()
    for i in range(len(weight)):
       bytes = struct.pack('f', weight[i])
       file.write(bytes)
    
    weight = model.fc[0].weight.detach()
    print('fc1')
    print(weight.max())
    print(weight.min())
    weight = weight.reshape(-1).numpy().tolist()
    for i in range(len(weight)):
       bytes = struct.pack('f', weight[i])
       file.write(bytes)
    
    weight = model.fc[2].weight.detach()
    print('fc2')
    print(weight.max())
    print(weight.min())
    weight = weight.reshape(-1).numpy().tolist()
    for i in range(len(weight)):
       bytes = struct.pack('f', weight[i])
       file.write(bytes)


if __name__ == "__main__":

    transform = transforms.Compose([transforms.ToTensor()])

    batch_size = 128
    dataset1 = datasets.MNIST('./data', train=True,
                              download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        dataset1, batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset2, 100, shuffle=True)

    model = NetworkC()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda")
    model = model.to(device)

    train(model, train_loader, device, batch_size)
    test(model, test_loader, device)

    save(model)
