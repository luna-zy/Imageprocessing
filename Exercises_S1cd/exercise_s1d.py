import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from collections import deque
import os


type Image = cv2.typing.MatLike

def get_kernel(size: int):
    l = 2*size+1
    return np.ones((l, l), np.uint8)
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
    

    print(f"Flat Zones Count: {label-1} (Connectivity={connectivity})")
    return label-1


def segment_coffee_grains(image_path):
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    markers = cv2.bitwise_not(img) # invert image
    _, markers = cv2.threshold(markers, 80, 255, cv2.THRESH_BINARY)

  # edit image to get markers
    markers = cv2.erode(markers, get_kernel(3))
    markers = cv2.morphologyEx(markers, cv2.MORPH_OPEN, get_kernel(4))
    markers = cv2.erode(markers, get_kernel(5))

    # find markers contours
    contours, _ = cv2.findContours(markers, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # output the markers in a white background
    cleaned_markers = np.ones((img.shape[0], img.shape[1]), np.uint8) * 255
    cv2.drawContours(cleaned_markers, contours, -1, 0, 7)

    # count the number of coffee grains
    num_coffee_grains = count_flat_zones_queue(8, cleaned_markers)    
    

    # 显示结果
    plt.figure(figsize=(12, 8))

    plt.subplot(2,1,1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title(f'Segmented Coffee Grains: {num_coffee_grains}')
    plt.subplot(2,1,2), plt.imshow(cleaned_markers, cmap='gray'), plt.title('Extracted Markers')
    plt.tight_layout()
    plt.show()

    return num_coffee_grains

# 运行分割函数并计数
image_path = 'TestImages/TestImages/coffee_grains.jpg'
num_coffee_grains = segment_coffee_grains(image_path)
print(f"Detected Coffee Grains: {num_coffee_grains}")
