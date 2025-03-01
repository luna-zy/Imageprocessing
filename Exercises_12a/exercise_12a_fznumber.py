import cv2
import numpy as np
import sys
from collections import deque

def flood_fill_queue(image, visited, label_map, x, y, label, connectivity):
    """ 使用队列（BFS）进行连通区域标记 """
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

def count_flat_zones_queue(input_txt, input_pgm, output_txt):
    """ 使用队列（BFS）计算输入 PGM 图像的平坦区域数量 """

    # 读取连通性参数（4 或 8）
    with open(input_txt, 'r') as f:
        connectivity = int(f.readline().strip())

    if connectivity not in [4, 8]:
        print("Error: Connectivity must be 4 or 8.")
        sys.exit(1)

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

    # 保存结果
    with open(output_txt, 'w') as f:
        f.write(str(label) + '\n')

    print(f"Flat Zones Count: {label} (Connectivity={connectivity})")
    print(f"Result saved to {output_txt}")

# 运行测试
input_txt = "Exercises_12a/exercise_12a_input_01.txt"
input_pgm = "Exercises_12a/immed_gray_inv.pgm"
output_txt = "Exercises_12a/exercise_12a_output_01.txt"

count_flat_zones_queue(input_txt, input_pgm, output_txt)
