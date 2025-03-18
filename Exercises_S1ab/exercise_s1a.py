import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_contour(image_path, kernel_size=3, operation='erosion'):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("无法读取输入图像，请检查路径")
    
    # 定义形态学核
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # 选择形态学操作
    if operation == 'erosion':
        morphed = cv2.erode(image, kernel, iterations=1)
    elif operation == 'dilation':
        morphed = cv2.dilate(image, kernel, iterations=1)
    else:
        raise ValueError("operation 只能是 'erosion' 或 'dilation'")

    # 计算轮廓
    contour = cv2.absdiff(image, morphed)

    # 显示结果
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')
    plt.subplot(1, 3, 2), plt.imshow(morphed, cmap='gray'), plt.title(f'{operation.capitalize()} Result')
    plt.subplot(1, 3, 3), plt.imshow(contour, cmap='gray'), plt.title('Extracted Contour')
    plt.show()

    return contour

# 示例调用
contour_img = extract_contour("TestImages/TestImages/hitchcock.png", kernel_size=3, operation='erosion')
