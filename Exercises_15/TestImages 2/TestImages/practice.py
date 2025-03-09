import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def manual_erode(image, kernel_size=3):
    """
    手动实现二值图像的侵蚀操作（符合公式 εB(I)）
    :param image: 二值化的 numpy 数组
    :param kernel_size: 结构元素的大小
    :return: 侵蚀后的图像 εB(I)
    """
    h, w = image.shape
    pad = kernel_size // 2  # 计算填充大小
    padded_img = np.pad(image, pad, mode='constant', constant_values=0)  # 填充边界

    eroded = np.zeros_like(image)  # 创建空白图像存放侵蚀结果

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8) * 255  # 结构元素

    for i in range(h):
        for j in range(w):
            region = padded_img[i:i+kernel_size, j:j+kernel_size]
            # 如果区域完全匹配 kernel（全255），则该像素保留
            if np.array_equal(region, kernel):
                eroded[i, j] = 255

    return eroded

def manual_subtract(imgA, imgB):
    """
    逐像素相减（符合公式 I' = I - εB(I)）
    :param imgA: 原图 I
    :param imgB: 侵蚀后的图 εB(I)
    :return: 边界提取图像 I'
    """
    return np.clip(imgA - imgB, 0, 255).astype(np.uint8)  # 保证像素值在 0-255

def boundary_extraction_manual(image_path, kernel_size=3):
    """
    手动实现边界提取（符合公式 I' = I - εB(I)）
    :param image_path: 输入图像路径
    :param kernel_size: 侵蚀核的大小
    :return: 原图 I, 侵蚀后图 εB(I), 边界提取图像 I'
    """
    # 读取灰度图像
    img = Image.open(image_path).convert("L")
    binary = np.array(img)

    # 二值化（符合公式 I）
    binary = np.where(binary > 128, 255, 0).astype(np.uint8)

    # 计算侵蚀（符合公式 εB(I)）
    eroded = manual_erode(binary, kernel_size)

    # 计算边界提取（符合公式 I' = I - εB(I)）
    boundary = manual_subtract(binary, eroded)

    return binary, eroded, boundary

# 测试代码
image_path = "hitchcock.png"  # 替换为你的图像路径
binary, eroded, boundary = boundary_extraction_manual(image_path)

# 显示结果
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(binary, cmap='gray')
ax[0].set_title("Original Binary Image (I)")
ax[0].axis("off")

ax[1].imshow(eroded, cmap='gray')
ax[1].set_title("Eroded Image (εB(I))")
ax[1].axis("off")

ax[2].imshow(boundary, cmap='gray')
ax[2].set_title("Extracted Boundary (I' = I - εB(I))")
ax[2].axis("off")

plt.show()