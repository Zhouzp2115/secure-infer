import torchvision
import struct
import torch
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as F


class NetworkA(nn.Module):
    def __init__(self):
        super(NetworkA, self).__init__()
        self.layer1 = nn.Linear(784, 128, bias=False)
        self.layer2 = nn.Linear(128, 128, bias=False)
        self.layer3 = nn.Linear(128, 10, bias=False)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        return x


def train(model, train_loader, device, batch_size):
    model.train()
    cost = torch.nn.CrossEntropyLoss()
    for epoch in range(20):
        print('train epoch ', epoch)
        for batch_idx, (data, target) in enumerate(train_loader):
           if((batch_idx + 1) * batch_size > 60000):
              break
           data, target = data.to(device), target.to(device)
           data = data.reshape(batch_size, -1)

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
            data = data.reshape(100, -1)
            output = F.softmax(model(data), 1)
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('\nTest set: Accuracy: {}/{} ({:.3f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def save(model):
    model = model.cpu()

    file = open('../bin/DATA/NetworkA-MNIST/weight.bin', 'wb')

    weight = model.layer1.weight.detach()
    weight = weight.reshape(-1).numpy().tolist()
    for i in range(len(weight)):
       bytes = struct.pack('f', weight[i])
       file.write(bytes)
    
    weight = model.layer2.weight.detach()
    weight = weight.reshape(-1).numpy().tolist()
    for i in range(len(weight)):
       bytes = struct.pack('f', weight[i])
       file.write(bytes)

    weight = model.layer3.weight.detach()
    weight = weight.reshape(-1).numpy().tolist()
    for i in range(len(weight)):
       bytes = struct.pack('f', weight[i])
       file.write(bytes)


if __name__ == "__main__":
    '''
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    '''
    transform = transforms.Compose([transforms.ToTensor()])

    batch_size = 128
    dataset1 = datasets.MNIST('./data', train=True,
                              download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        dataset1, batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset2, 100, shuffle=True)

    model = NetworkA()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda")
    model = model.to(device)
    
    train(model, train_loader, device, batch_size)
    test(model, test_loader, device)

    print('layer1')
    print(model.layer1.weight.max())
    print(model.layer1.weight.min())
    print('layer2')
    print(model.layer2.weight.max())
    print(model.layer2.weight.min())
    print('layer3')
    print(model.layer3.weight.max())
    print(model.layer3.weight.min())

    save(model)

