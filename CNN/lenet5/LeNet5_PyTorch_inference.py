# # OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/.
# import os
#
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# import torch
# from PIL import Image
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# from CNN.LeNet5_PyTorch import LeNet5
# import torch.nn.functional as F
#
# # 加载模型
# model = LeNet5()
# model.load_state_dict(torch.load("LeNet5_PyTorch.pth"))
# model.eval()  # 设置为评估模式
#
#
#
# # --- 1. 定义一个与训练时完全一致的预处理流程 ---
# # 注意：如果你的原始图片是黑底白字，请注释掉 InvertColor() 这一行
# class InvertColor(object):
#     def __call__(self, tensor):
#         return F.invert(tensor)
#
# # 定义数据变换  要和训练用的数据、数据转换操作 完全一致
# transform = transforms.Compose(
#     [
#         transforms.Grayscale(),
#         transforms.Resize((28, 28)),
#         transforms.ToTensor(),
#         InvertColor(),  # 如果你的图片是白底黑字，则保留此行
#         # transforms.Normalize((0.1307,), (0.3081,))
#         transforms.Normalize((0.5,), (0.5,))
#         # Normalize 的参数完全不同！训练时模型看到的是被归一化到 [-1, 1] 区间的数据，而预测时你喂给它的数据是用另一套均值和标准差归一化的。模型自然无法理解
#     ]
# )
#
# # 加载并处理自己的图片
# image = Image.open("bigcu6.png")  # 替换为你的图片路径
# image = transform(image).unsqueeze(0)  # 添加批次维度
# # 转换为PIL图像
# transformed_pil = transforms.ToPILImage()(image.squeeze())
#
# # 显示对比图
# plt.figure(figsize=(12, 6))
#
# # 归一化后图片
# plt.subplot(1, 3, 2)
# plt.title("Normalized Image")
# plt.imshow(transformed_pil, cmap='gray')  # 灰度显示
#
# plt.show()
#
# # 预测
# with torch.no_grad():
#     output = model(image)
#     _, predicted = torch.max(output.data, 1)
#     print(f"预测结果: {predicted.item()}")
# ---------------------------------------------------------------------

# OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/.
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
# 确保这里引入的是你训练好的模型类
from CNN.lenet5.LeNet5_PyTorch import LeNet5

# --- 1. 定义一个与训练时完全一致的预处理流程 ---
# 注意：如果你的原始图片是黑底白字，请注释掉 InvertColor() 这一行
class InvertColor(object):
    def __call__(self, tensor):
        return F.invert(tensor)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    # InvertColor(), # 如果你的图片是白底黑字，则保留此行
    transforms.Normalize((0.5,), (0.5,))
])

# --- 2. 加载模型 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet5().to(device)
model.load_state_dict(torch.load("LeNet5_PyTorch.pth", map_location=device))
model.eval()  # 设置为评估模式

# --- 3. 加载并处理你的图片 ---
try:
    image_raw = Image.open("3.png") #
except FileNotFoundError:
    print("错误：请将 'your_digit_image.png' 替换为你的手写数字图片路径！")
    exit()

image = transform(image_raw).unsqueeze(0).to(device) # 添加批次维度并移动到设备

# --- 4. 可视化检查（关键步骤） ---
# 我们需要反归一化才能正常显示图片
def imshow(tensor):
    # 反归一化: tensor * 0.5 + 0.5
    image = tensor.cpu().clone().squeeze()
    image = image * 0.5 + 0.5
    plt.imshow(image, cmap='gray')
    plt.title("Preprocessed Image (What the Model Sees)")
    plt.show()

print("显示预处理后的图片，检查它是否像一张标准的MNIST图片（黑底白字）...")
imshow(image)


# --- 5. 预测 ---
with torch.no_grad():
    output = model(image)
    probabilities = torch.softmax(output, dim=1)
    confidence, predicted = torch.max(probabilities, 1)

    print(f"\n预测结果: {predicted.item()}")
    print(f"置信度: {confidence.item() * 100:.2f}%")