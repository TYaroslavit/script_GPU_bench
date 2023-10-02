import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from timeit import default_timer as timer

start = timer()
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

batch_size = 32
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

GPU_available = torch.cuda.is_available()
print(f"GPU available? {GPU_available}")
numGPUs = torch.cuda.device_count()
print(f"Number of GPUs available: {numGPUs}")
FIRST_GPU = "cuda:0"
device = torch.device(FIRST_GPU if GPU_available else "cpu")
print(f"Device: {device}")
deviceIndex = torch.cuda.current_device()
print(f"Index of currently selected device: {deviceIndex}")
print(f"Device name: {torch.cuda.get_device_name(deviceIndex)}")
#torch.cuda.set_device(deviceIndex)

class CIFAR10Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.flat = nn.Flatten()

        self.fc3 = nn.Linear(8192, 512)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(512, 10)

    def forward(self, x):
        # input 3x32x32, output 32x32x32
        x = self.act1(self.conv1(x))
        x = self.drop1(x)
        # input 32x32x32, output 32x32x32
        x = self.act2(self.conv2(x))
        # input 32x32x32, output 32x16x16
        x = self.pool2(x)
        # input 32x16x16, output 8192
        x = self.flat(x)
        # input 8192, output 512
        x = self.act3(self.fc3(x))
        x = self.drop3(x)
        # input 512, output 10
        x = self.fc4(x)
        return x

model = CIFAR10Model(); model = model.to(device)
loss_fn = nn.CrossEntropyLoss(); loss_fn = loss_fn.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) #; optimizer = optimizer.to(device)

n_epochs = 20
for epoch in range(n_epochs):
    for inputs, labels in trainloader:
        inputs = inputs.to(device); labels = labels.to(device)
        # forward, backward, and then weight update
        y_pred = model(inputs)
        loss = loss_fn(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    acc = 0
    count = 0
    for inputs, labels in testloader:
        inputs = inputs.to(device); labels = labels.to(device)
        y_pred = model(inputs)
        acc += (torch.argmax(y_pred, 1) == labels).float().sum()
        count += len(labels)
    acc /= count
    print("Epoch %d: model accuracy %.2f%%" % (epoch, acc*100))

end = timer()
print(f"Time taken in minutes: {(end - start) / 60}") # Time in seconds
torch.save(model.state_dict(), "cifar10model.pth")
