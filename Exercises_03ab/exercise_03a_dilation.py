import cv2
import numpy as np

def dilation(image, kernel_size):
    # 创建一个正方形的结构元素
    kernel = np.ones((2 * kernel_size + 1, 2 * kernel_size + 1), np.uint8)
    # 使用cv2.erode进行腐蚀操作
    eroded_image = cv2.dilate(image, kernel, iterations=1)
    return eroded_image

image = cv2.imread("Exercises_03ab/immed_gray_inv.pgm", cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: Could not open or find the image.")
    
else:
    print("Input image loaded successfully.")

output_file = "Exercises_03ab/immed_gray_inv_out2.pgm"
eroded_image = dilation(image, 2)
print("Erosion completed.")
cv2.imwrite(output_file, eroded_image)
print(f"Output image saved to {output_file}")