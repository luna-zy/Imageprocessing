"""Exercise 02d  In this exercise we are going to compare 
the number of operations in two alternatives 
for computing a morphological dilation with structuring element.
Let B be the MxM square structuring element.
Let C be the 1xM 1-D horizontal structuring element:
Let D be the Mx1 1-D vertical structuring elemen"""

"""Note:    − The number of pixels of B is MxM      
            − The number of pixels of C and D is M.
'X' denotes the origin of coordinates or center of the structuring element. 
B, C and D are centered structuringelements.

It can be observed that the following property holds:
B = dilate_C (D) = dilate_D (C).
Estimate the number or 'max' operations that must be computed in
order to process a NxN square input image using the following
alternatives:
dilate_B (I))
dilate_C(dilate_D (I)))

Border effects should not be considered for simplicity, i.e.,
all image pixels should be treated in the same manner"""

"""该练习用于  比较两种形态学膨胀的操作数
    B是MxM的方形结构元素
    C是1xM的1-D水平结构元素
    D是Mx1的1-D垂直结构元素
    B = dilate_C (D) = dilate_D (C)
    估计在处理NxN正方形输入图像时必须计算的最大操作数
    使用以下备选方案：
    dilate_B (I))
    dilate_C(dilate_D (I)))
    为简单起见，不应考虑边界效应，即所有图像像素应以相同方式处理"""

# public libraries
import sys
import os
import cv2
import numpy as np
# prviate libraries
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Exercises_02ab.exercise_02b_compare import exercise_02b_compare
from Exercises_06ab.exercise_06a_closing_opening import custom_dilate
##
import time

def dilate_B(image, kernel_size):
    """ 直接使用 MxM 结构元素进行膨胀 """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

def dilate_C_D(image, kernel_size):
    """ 先用 1D 结构元素 D (Mx1) 进行膨胀，再用 C (1xM) 进行膨胀 """
    kernel_D = np.ones((kernel_size, 1), np.uint8)  # Mx1
    kernel_C = np.ones((1, kernel_size), np.uint8)  # 1xM

    temp = cv2.dilate(image, kernel_D, iterations=1)  # 先用 D
    result = cv2.dilate(temp, kernel_C, iterations=1)  # 再用 C
    return result

def count_operations(image, kernel_size, method="B"):
    """ 计算 max 操作次数 """
    h, w = image.shape
    if method == "B":
        # 直接 B (MxM) 计算量 = N² * M²
        return h * w * (kernel_size ** 2)
    elif method == "C_D":
        # 先 D (Mx1) + 再 C (1xM) 计算量 = N² * M + N² * M = 2 * N² * M
        return h * w * kernel_size * 2
    else:
        return 0

# 运行测试
input_file = "Exercises_01a/cam_74.pgm"
kernel_size = 5  # M=5
image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Unable to read the image.")
    sys.exit(1)

# 计算直接 B 膨胀的时间和计算量
start_time = time.time()
dilated_B = dilate_B(image, kernel_size)
time_B = time.time() - start_time
ops_B = count_operations(image, kernel_size, method="B")

# 计算 C-D 分解的时间和计算量
start_time = time.time()
dilated_C_D = dilate_C_D(image, kernel_size)
time_C_D = time.time() - start_time
ops_C_D = count_operations(image, kernel_size, method="C_D")

# 保存结果
cv2.imwrite("Exercises_02d/dilated_B.pgm", dilated_B)
cv2.imwrite("Exercises_02d/dilated_C_D.pgm", dilated_C_D)

# 结果输出
print(f"Direct B(I) - Time: {time_B:.5f}s, Operations: {ops_B}")
print(f"Separated C-D - Time: {time_C_D:.5f}s, Operations: {ops_C_D}")

# 显示图像
cv2.imshow("Original", image)
cv2.imshow("Dilated B", dilated_B)
cv2.imshow("Dilated C-D", dilated_C_D)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''important!!!'''
"""直接B和先C再D是一样的嘛,但是操作数少了
直接使用 B 进行膨胀 和 先 C 再 D 进行膨胀 在数学上是等价的，但计算量（操作数）减少了，这是因为 C-D 方法利用了分解的优势。
直观理解：
B 是一个 M*M 的方形结构元素，需要计算 M² 个像素的最大值。
C 是 1*M, D 是 M*1, 它们分别计算 M 个像素的最大值。
🚀 关键点

先 C 再 D(或者先 D 再 C)得到的结果，数学上与直接 B 一样，但计算量更少！
C-D 方法比 B 方法快 2-3 倍！
图像结果相同，但 C-D 方法减少了计算量！
"""