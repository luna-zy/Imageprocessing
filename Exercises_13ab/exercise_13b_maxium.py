import numpy as np
import cv2
import sys
from collections import deque

def read_input_file(filename):
    with open(filename, 'r') as f:
        x = int(f.readline().strip())
        y = int(f.readline().strip())
        connectivity = int(f.readline().strip())
        flat_zone_label = int(f.readline().strip())
    return x, y, connectivity, flat_zone_label

def read_pgm_image(filename):
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    return image

def get_neighbors(p, shape, connectivity):
    x, y = p
    neighbors = []
    
    if connectivity == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:  # 8-connectivity
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    for dx, dy in deltas:
        nx, ny = x + dx, y + dy
        if 0 <= nx < shape[0] and 0 <= ny < shape[1]:
            neighbors.append((nx, ny))
    
    return neighbors

def compute_flat_zone(image, x, y, connectivity, flat_zone_label):
    h, w = image.shape
    output = np.zeros((h, w), dtype=np.uint8)
    label_no_fz = 0  # Default non-flat zone label
    
    seed_value = image[y, x]
    queue = deque([(y, x)])
    output[y, x] = flat_zone_label
    
    while queue:
        cy, cx = queue.popleft()
        
        for ny, nx in get_neighbors((cy, cx), (h, w), connectivity):
            if output[ny, nx] == label_no_fz and image[ny, nx] == seed_value:
                output[ny, nx] = flat_zone_label
                queue.append((ny, nx))
    
    return output
def is_regional_maximum(image, flat_zone):
    h, w = image.shape
    
    # 获取 flat zone 里的像素值集合
    flat_zone_pixels = image[flat_zone == 255]
    
    # 计算 flat zone 内的最大像素值
    max_value = np.max(flat_zone_pixels)

    for y in range(h):
        for x in range(w):
            if flat_zone[y, x] == 255:  # 仅检查 flat zone 内的像素
                for ny, nx in get_neighbors((y, x), (h, w), 8):  # 8-邻域
                    if flat_zone[ny, nx] != 255:  # 只检查 flat zone 以外的邻居
                        if image[ny, nx] >= max_value:  # 若邻居值 >= flat zone 最大值
                            #print(f"Checking ({ny}, {nx}): {image[ny, nx]} vs {max_value}")
                            return 0  # 不是区域最大值
    return 1  # 是区域最大值

    
def write_output_text(filename, value):
    with open(filename, 'w') as f:
        f.write(str(value) + "\n")

def main():
    input_image="Exercises_13ab/immed_gray_inv_20051218_frgr4.pgm"
    output_pgm="Exercises_13ab/exercise_13b_output_01.pgm"
    output_txt="Exercises_13ab/exercise_13b_output_01.txt"
    input_txt="Exercises_13ab/exercise_13b_input_01.txt"

    x, y, connectivity, flat_zone_label = read_input_file(input_txt)
    image = read_pgm_image(input_image)
    output= compute_flat_zone(image, x, y, connectivity, flat_zone_label)
    cv2.imwrite(output_pgm, output)
    reg_max = is_regional_maximum(image, output)
    write_output_text(output_txt, reg_max)
    print("reg_max=", reg_max)
    print("Flat zone computed and saved to", output_txt)

    
    
if __name__ == "__main__":
    main()

