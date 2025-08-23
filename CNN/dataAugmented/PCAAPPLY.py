import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from CNN.dataAugmented.PCAColorAugmentation import PCAColorAugmentation


# 定义数据转换流程
# 通常在转换为Tensor之前应用PCA增强
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    PCAColorAugmentation(), # 在这里使用
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 创建数据集
train_dataset = ImageFolder(root='path/to/your/data', transform=train_transforms)