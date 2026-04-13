from torchvision.datasets.mnist import MNIST
import torch
from torch import nn

train_mnist = MNIST('./data/mnist', train=True, download=True)
test_mnist = MNIST('./data/mnist', train=False, download=True)

def data_gen(x, y, batch_size=256):
    while True:
        idx = torch.randint(low=0, high=len(x), size=(batch_size,), device='cuda') # On gpu
        yield x[idx], y[idx]

train_gen = data_gen(
    train_mnist.data.cuda(), # This will copy it on gpu
    train_mnist.targets.cuda()
)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 4, (5, 5), padding=0),
            nn.ReLU(),
            nn.Conv2d(4, 4, (5, 5), padding=0),
            nn.ReLU(),
            nn.Conv2d(4, 8, (5, 5), padding=0),
            nn.ReLU(),
            nn.Conv2d(8, 16, (5, 5), padding=0),
            nn.ReLU()
        )

        self.fc = nn.Linear(2304, 10)

    def forward(self, x):
        net = x[:, None].type(torch.float32)
        net = self.backbone(net)
        net = net.view(-1, 16 * 12 * 12)
        net = self.fc(net)
        # net = nn.functional.relu()
        return net



model = Net()
model = model.cuda() # move it on GPU

# params_playlist = model.parameters()
# print(next(params_playlist).size()) 
# print(next(params_playlist).size()) 
# print(next(params_playlist).size())
# print(next(params_playlist).size())

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

def train_step(x, y):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()

for step in range(1000):
    loss = train_step(*next(train_gen))
    if step % 100 == 0:
        with torch.no_grad():
            y_test_pred = model(test_mnist.data.cuda())
            accuracy = (y_test_pred.argmax(axis=1).cpu() == test_mnist.targets).type(torch.float32).mean()

        print(f"\nStep: {step} | Loss: {loss} | Accuracy: {accuracy}")
    
# print(model)
