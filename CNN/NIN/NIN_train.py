import torch
from torch import Tensor, nn, optim

from CNN.alexnet.training_tools import fashionMNIST_loader, Trainer
from CNN.NIN.NIN import NiN





if __name__ == '__main__':
    BATCH_SIZE = 128
    EPOCHS_NUM = 30
    LEARNING_RATE = 0.01

    model = NiN(num_classes=10)
    train_loader, test_loader = fashionMNIST_loader(BATCH_SIZE, resize=224)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), LEARNING_RATE)
    platform = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with Trainer(model, train_loader, test_loader, criterion, optimizer, platform) as trainer:
        trainer.train(EPOCHS_NUM,"nin4fashionmnist.pth")  # 记得修改保存的模型名字  懒得该代码目前