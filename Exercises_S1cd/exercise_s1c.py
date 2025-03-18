import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from collections import deque
import os




def flood_fill_queue(image, visited, label_map, x, y, label, connectivity):

    queue = deque([(x, y)])
    h, w = image.shape
    gray_value = image[x, y]  # 当前区域的灰度值

    # 定义邻居（8-连通 或 4-连通）
    if connectivity == 8:
        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),         (0, 1),
                     (1, -1), (1, 0), (1, 1)]
    else:  # 4-连通性
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        cx, cy = queue.popleft()
        if visited[cx, cy]:  # 如果已访问，跳过
            continue

        visited[cx, cy] = True  # 标记访问
        label_map[cx, cy] = label  # 赋予新标签

        # 遍历邻居
        for dx, dy in neighbors:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < h and 0 <= ny < w:
                if not visited[nx, ny] and image[nx, ny] == gray_value:
                    queue.append((nx, ny))  # 加入队列，继续搜索

def count_flat_zones_queue(input_txt, input_pgm):


    # 读取连通性参数（4 或 8）
    # if input_txt is a file, read the connectivity from the file
    if os.path.isfile(input_txt):
    
        with open(input_txt, 'r') as f:
            connectivity = int(f.readline().strip())
    # if input_txt is a integer, use the integer as the connectivity
    else:
        connectivity = int(input_txt)

    if connectivity not in [4, 8]:
        print("Error: Connectivity must be 4 or 8.")
        sys.exit(1)

    # if input_pgm has been read, use the image
    if isinstance(input_pgm, np.ndarray):
        image = input_pgm

    else:
    # 读取 PGM 图像
        image = cv2.imread(input_pgm, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Unable to read the image.")
        sys.exit(1)

    h, w = image.shape
    visited = np.zeros((h, w), dtype=bool)  # 访问标记
    label_map = np.zeros((h, w), dtype=np.int32)  # 记录连通区域
    label = 0  # 当前标签计数

    # 遍历图像像素
    for i in range(h):
        for j in range(w):
            if not visited[i, j]:  # 如果未访问过
                label += 1  # 分配新标签
                flood_fill_queue(image, visited, label_map, i, j, label, connectivity)


    print(f"Flat Zones Count: {label} (Connectivity={connectivity})")

def get_kernel(size: int):
    l = 2*size+1
    return np.ones((l, l), np.uint8)


def count_wheel_teeth(image_path):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("could not read input image, please check the path")
    _, image = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY) # threshold image to binary

    # 进行高斯模糊，减少噪声
    dilatedImg = cv2.dilate(image, get_kernel(1))
    erodedImg = cv2.erode(dilatedImg, get_kernel(5))
    teeth = image - erodedImg


    # 轮廓检测
    contours, _ = cv2.findContours(teeth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    teeth.fill(0)
    cv2.drawContours(teeth, contours, -1, 255, 1)   
    teeth = cv2.dilate(teeth, get_kernel(2))
    teeth = cv2.erode(teeth, get_kernel(2))
    otherimg = cv2.dilate(image, get_kernel(3))
    otherimg = cv2.erode(otherimg, get_kernel(4))
    otherimg = cv2.morphologyEx(otherimg, cv2.MORPH_OPEN, get_kernel(10))

    teeth = cv2.subtract(teeth, otherimg) 


    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')
    plt.subplot(1, 2, 2), plt.imshow(teeth, cmap='gray'), plt.title('Extracted Teeth')
    plt.show()


    flat_zones = count_flat_zones_queue(8, teeth)
    

    return flat_zones

# 示例调用

wheel_teeth = count_wheel_teeth("TestImages/TestImages/wheel.png")
