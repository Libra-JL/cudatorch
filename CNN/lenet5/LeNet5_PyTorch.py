import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 定义数据变换
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
        transforms.Normalize((0.5,), (0.5,))
    ]
)

# 加载数据
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 准备输入数据（假设是一个MNIST图像数据）
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# 定义模型
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.pool1_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fully_connected1_5 = nn.Linear(in_features=16 * 4 * 4, out_features=120)
        self.fully_connected2_6 = nn.Linear(in_features=120, out_features=84)
        self.fully_connected3_7 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.pool1_2(torch.relu(self.conv1_1(x)))
        x = self.pool2_4(torch.relu(self.conv2_3(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fully_connected1_5(x))
        x = torch.relu(self.fully_connected2_6(x))
        x = self.fully_connected3_7(x)
        return x


# 初始化 lenet-5 模型以及定义损失函数和优化器
device = torch.device(torch.accelerator.current_accelerator() if torch.cuda.is_available() else "cpu")
print(device)
net = LeNet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

if __name__ == '__main__':
    # 定义训练
    for epoch in range(20):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

    torch.save(net.state_dict(), "LeNet5_PyTorch.pth")
    print("Finished Training")

#
#
# # 测试模型
# correct = 0
# total = 0
# with torch.no_grad():
#     for data in test_loader:
#         inputs, labels = data
#         inputs = inputs.to(device)
#         labels = labels.to(device)
#         outputs = net(inputs)  # 前向传播
#         _, predicted = torch.max(outputs.data, 1)  # 找到最大概率的类别
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# accuracy = 100 * correct / total
# print(f"Accuracy on the test set: {accuracy}%")