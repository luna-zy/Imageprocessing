# public libraries 
import sys
import os
import cv2
import numpy as np
# prviate libraries
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Exercises_13ab.exercise_13a_minimun import read_input_file, read_pgm_image, get_neighbors, compute_flat_zone, is_regional_minimum
from Exercises_13ab.exercise_13b_maxium import is_regional_maximum
##
def compute_regional_extrema(image, extrema_type="min"):
    """计算整个图像中的所有区域最值"""
    h, w = image.shape
    output = np.zeros((h, w), dtype=np.uint8)

    visited = np.zeros((h, w), dtype=bool)  # 记录哪些像素已经处理过

    for y in range(h):
        for x in range(w):
            if not visited[y, x]:  # 只处理未访问的像素
                flat_zone = compute_flat_zone(image, x, y, 8, 255)
                visited[flat_zone == 255] = True  # 标记已访问

                if extrema_type == "min" and is_regional_minimum(image, flat_zone):
                    output[flat_zone == 255] = 255  # 区域最小值设为 255
                elif extrema_type == "max" and is_regional_maximum(image, flat_zone):
                    output[flat_zone == 255] = 255  # 区域最大值设为 255

    return output
if __name__ == "__main__":
    image = read_pgm_image("Exercises_13cd/immed_gray_inv_20051218_frgr4.pgm")
    output = compute_regional_extrema(image, extrema_type="min")
    cv2.imwrite("Exercises_13cd/exercise_13c_output_01.pgm", output)

