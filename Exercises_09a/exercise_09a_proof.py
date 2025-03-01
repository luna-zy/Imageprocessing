import cv2
import numpy as np
import sys

# 形态学 Closing-Opening 交替滤波
def closing_opening(image, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    opened_closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened_closed

# 读取图像
input_file = "Exercises_04ab/immed_gray_inv.pgm"
image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)

# 计算一次 Closing-Opening
result_1 = closing_opening(image, 5)

# 计算两次 Closing-Opening
result_2 = closing_opening(result_1, 5)

# 计算像素差异
difference = np.abs(result_1.astype(np.int16) - result_2.astype(np.int16))
diff_sum = np.sum(difference)

# 结果检查
if diff_sum == 0:
    print("Closing-Opening is idempotent: result_1 == result_2")
else:
    print(f"Closing-Opening is NOT idempotent: total difference {diff_sum}")

# 显示图像
cv2.imshow("Original", image)
cv2.imshow("Closing-Opening Once", result_1)
cv2.imshow("Closing-Opening Twice", result_2)
cv2.waitKey(0)
cv2.destroyAllWindows()
