import cv2 
import sys
import os

def compare(input_file1, input_file2):
    """ 比较两幅 PGM 图像是否相同 """
    image1 = cv2.imread(input_file1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(input_file2, cv2.IMREAD_GRAYSCALE)
    h1, w1 = image1.shape
    h2, w2 = image2.shape
    flag = 0  # 默认图像不同

    if (h1, w1) == (h2, w2):  # 如果尺寸相同
        if (image1 == image2).all():  # 判断所有像素是否相同
            flag = 1  # 设为相同

    return flag
import os

import os

def exercise_02b_compare(input_file1, input_file2, output_file=None):
    """ 
    比较两幅 PGM 图像是否相同
    - 如果 `output_file` 为空，则按照 `[当前文件夹名]_output_i.txt` 格式存储
    - 如果已有相同命名的 `output_XX.txt`，则自动递增 `i`
    """

    i = 1  # 计数编号

    # 获取当前 `.py` 文件所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 获取当前 `.py` 文件所在的文件夹名
    folder_name = os.path.basename(current_dir)

    # 如果 `output_file` 为空，则按 `[文件夹名]_output_i.txt` 格式命名
    if output_file is None:
        output_file = os.path.join(current_dir, f"{folder_name}_output_{i:02d}.txt")
    else:
        # 拆分提供的 `output_file` 的文件名和扩展名
        base_name, ext = os.path.splitext(output_file)

    # 查找下一个可用的 `output_X.txt`
    while os.path.exists(output_file):
        i += 1
        output_file = os.path.join(current_dir, f"{folder_name}_output_{i:02d}.txt")

    # 这里假设 `compare()` 是一个函数，比较两张图像，返回 `1` 表示相同，`0` 表示不同
    flag = compare(input_file1, input_file2)

    # 将结果写入 `output_file`
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(str(flag))

    # 打印比较结果
    if flag == 1:
        print("Images are identical.") 
    else:
        print("Images are different.")

    return flag

if __name__ == "__main__":

    # 读取两张 PGM 图像

    image1_path = "Exercises_02ab/cam_74_threshold100.pgm"
    image2_path = "Exercises_02ab/cam_74.pgm"
    output_file= "Exercises_02ab/exercise_02b_output_01.txt"
    exercise_02b_compare(image1_path, image2_path, output_file)
    
    image1=cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2=cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    # imshow()
    cv2.imshow("image1", image1)
    cv2.imshow("image2", image2)

    cv2.waitKey(0)  # wait for a key press to close the window
    cv2.destroyAllWindows()  # close all windows