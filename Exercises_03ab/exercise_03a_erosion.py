import cv2
import numpy as np

def erosion(image, kernel_size):
    # 创建一个正方形的结构元素
    kernel = np.ones((2 * kernel_size + 1, 2 * kernel_size + 1), np.uint8)
    # 使用cv2.erode进行腐蚀操作
    eroded_image = cv2.erode(image, kernel, iterations=1)
    return eroded_image

image = cv2.imread("Exercises_03ab/immed_gray_inv.pgm", cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: Could not open or find the image.")
    
else:
    print("Input image loaded successfully.")

output_file = "Exercises_03ab/immed_gray_inv_out.pgm"
eroded_image = erosion(image,2)
print("Erosion completed.")
cv2.imwrite(output_file, eroded_image)
print(f"Output image saved to {output_file}")



'''
def main():
    if len(sys.argv) != 4:
        print("Usage: python exercise_03a_erosion.py i input.pgm output.pgm")
        print(f"Provided arguments: {sys.argv}")  # 打印实际提供的参数
        return

    i = 1 #int(sys.argv[1])
    input_file = "Exercises_03ab/immed_gray_inv.pgm"
    output_file = "Exercises_03ab/immed_gray_inv_out.pgm"

    print(f"Running erosion with i={1}, input_file={input_file}, output_file={output_file}")

    # 读取图像
    image = cv2.imread("Exercises_03ab/immed_gray_inv.pgm", cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Could not open or find the image.")
        return
    else:
        print("Input image loaded successfully.")

    # 进行腐蚀操作
    eroded_image = erosion(image, i)
    print("Erosion completed.")

    # 保存输出图像
    cv2.imwrite(output_file, eroded_image)
    print(f"Output image saved to {output_file}")




#if __name__ == "__main__":
    main()'''