import cv2 

# 读取两张 PGM 图像
image1 = cv2.imread("/Users/luna/Documents/UPM_course/Image processing/Exercises/Exercises_02ab/cam_74_threshold100.pgm", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("/Users/luna/Documents/UPM_course/Image processing/Exercises/Exercises_02ab/cam_74.pgm", cv2.IMREAD_GRAYSCALE)

# 获取图像尺寸
h1, w1 = image1.shape
h2, w2 = image2.shape

flag = 0  # 默认图像不同

if (h1, w1) == (h2, w2):  # 如果尺寸相同
    if (image1 == image2).all():  # 判断所有像素是否相同
        flag = 1  # 设为相同

# 输出结果
print(flag)

# 将结果写入文件
with open("Exercises_02ab/exercise_02b_output_01.txt", "w") as f:
    f.write(str(flag))
















def read_pgm(filename):
    with open(filename, 'rb') as f:
        header = f.readline().decode().strip()
        if header not in ('P2', 'P5'):
            return None

        while True:
            line = f.readline().decode().strip()
            if not line.startswith("#"):
                width, height = map(int, line.split())
                break

        f.readline()  # 跳过最大灰度值
        pixels = [int(v) for line in f for v in line.split()] if header == 'P2' else list(f.read())

    return width, height, pixels

def compare_pgm(file1, file2):
    img1 = read_pgm(file1)
    img2 = read_pgm(file2)
    return '1' if img1 and img2 and img1 == img2 else '0'
#
#if __name__ == "__main__":
#    if len(sys.argv) == 3:
#        result = compare_pgm(sys.argv[1], sys.argv[2])
#       with open("exercise_02b_output_01.txt", "w") as f:
#            f.write(result)