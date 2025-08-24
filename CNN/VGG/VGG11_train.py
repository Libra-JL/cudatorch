import torch
from torch import nn, optim
from VGG11 import VGG11
from CNN.alexnet.training_tools import fashionMNIST_loader, Trainer

if __name__ == '__main__':
    BATCH_SIZE = 128
    EPOCHS_NUM = 30
    LEARNING_RATE = 0.05

    model = VGG11(num_classes=10)
    train_loader, test_loader = fashionMNIST_loader(BATCH_SIZE, resize=224)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), LEARNING_RATE)
    platform = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with Trainer(model, train_loader, test_loader, criterion, optimizer, platform) as trainer:
        trainer.train(EPOCHS_NUM)
