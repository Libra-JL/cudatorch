import numpy as np
from PIL import Image


class PCAColorAugmentation(object):
    """
    对图像进行PCA颜色增强，也称为Fancy PCA。
    这个类实现了AlexNet论文中描述的数据增强方法。
    """

    def __init__(self):
        # 这些是基于 ImageNet 数据集计算得出的近似值
        # 特征值 (eigenvalues)
        self.eigenvalues = np.array([0.2175, 0.0188, 0.0045])
        # 特征向量 (eigenvectors)，每一列是一个特征向量
        self.eigenvectors = np.array([
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ]).T  # 转置使得每一行是一个特征向量，方便计算

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): 输入的PIL图像.

        Returns:
            PIL.Image: 经过PCA颜色增强后的图像。
        """
        # 1. 将PIL图像转换为numpy数组，并归一化到[0, 1]
        img_np = np.array(img, dtype=np.float32) / 255.0

        # 2. 生成随机变量 beta，符合 N(0, 0.1)
        # alpha 的形状是 (3,)
        betas = np.random.normal(0, 0.1, 3)

        # 3. 计算扰动量
        # (3,) * (3,) -> (3,)
        alphas = betas * self.eigenvalues

        # 4. 将扰动量投影到颜色空间
        # (3, 3) @ (3,) -> (3,)
        # 这就是公式 [p1, p2, p3] * [β1λ1, β2λ2, β3λ3]^T 的实现
        rgb_offset = self.eigenvectors @ alphas

        # 5. 将计算出的偏移量添加到图像的每个像素上
        # img_np 的形状是 (H, W, 3)
        # rgb_offset 的形状是 (3,)
        # NumPy的广播机制会自动将offset加到每个像素上
        img_aug = img_np + rgb_offset

        # 6. 将像素值裁剪到[0, 1]范围
        img_aug = np.clip(img_aug, 0, 1)

        # 7. 转换回 uint8 格式并变回 PIL.Image
        img_aug = (img_aug * 255).astype(np.uint8)

        return Image.fromarray(img_aug)


# --- 使用示例 ---
if __name__ == '__main__':
    try:
        # 加载一张示例图片
        input_image = Image.open("img.png").convert("RGB")

        # 创建增强实例
        color_augmenter = PCAColorAugmentation()

        # 应用增强
        augmented_image = color_augmenter(input_image)

        # 显示或保存结果
        input_image.show(title="Original Image")
        augmented_image.show(title="Augmented Image")

        # 保存增强后的图像
        augmented_image.save("augmented_image.jpg")
        print("增强后的图像已保存为 augmented_image.jpg")

    except FileNotFoundError:
        print("请将'your_image.jpg'替换为你的图片文件路径。")