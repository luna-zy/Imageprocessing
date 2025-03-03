import numpy as np
import cv2
import sys
from collections import deque
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Exercises_02ab.exercise_02b_compare import exercise_02b_compare

def read_input_file(filename):
    with open(filename, 'r') as f:
        x = int(f.readline().strip())
        y = int(f.readline().strip())
        connectivity = int(f.readline().strip())
        flat_zone_label = int(f.readline().strip())
    return x, y, connectivity, flat_zone_label


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

def compute_flat_zone(image, x, y, connectivity, LABEL_FZ):
    h, w = image.shape
    output = np.zeros_like(image)
    original_value = image[x, y]
    queue = deque([(x, y)])
    output[x, y] = LABEL_FZ
    LABEL_NO_FZ = 0  # Default non-flat zone label
    
    while queue:
        cx, cy = queue.popleft()
        
        for nx, ny in get_neighbors((cy, cx), (h, w), connectivity):
            if output[nx, ny] == LABEL_NO_FZ and image[ny, nx] == original_value:
                output[nx, ny] = LABEL_FZ
                queue.append((nx, ny))
    
    return output


if __name__ == "__main__":    

    input_txt1= "Exercises_11a/exercise_11a_input_01.txt"
    input_txt2= "Exercises_11a/exercise_11a_input_02.txt"
    input_pgm1= "Exercises_11a/gran01_64.pgm"
    input_pgm2= "Exercises_11a/immed_gray_inv_20051218_frgr4.pgm"
    output_pgm1="Exercises_11a/exercise_11a_outpu_01.pgm"
    output_pgm2="Exercises_11a/exercise_11a_outpu_02.pgm"
    output_txt1="Exercises_11a/exercise_11a_output_01.txt"
    output_txt2="Exercises_11a/exercise_11a_output_02.txt"
    compare_pgm1="Exercises_11a/gran01_64_flatzone0_0.pgm"
    compare_pgm2="Exercises_11a/immed_gray_inv_20051218_frgr4_flatzone57_36.pgm"

    x1, y1, connectivity1, LABEL_FZ1 = read_input_file(input_txt1)
    image1 = cv2.imread(input_pgm1, cv2.IMREAD_GRAYSCALE)
    output1 = compute_flat_zone(image1, x1, y1, connectivity1, LABEL_FZ1)
    cv2.imwrite(output_pgm1, output1)
    exercise_02b_compare(input_pgm1, output_pgm1, output_txt1)

    x2, y2, connectivity2, LABEL_FZ2 = read_input_file(input_txt2)
    image2 = cv2.imread(input_pgm2, cv2.IMREAD_GRAYSCALE)
    output2 = compute_flat_zone(image2, x2, y2, connectivity2, LABEL_FZ2)
    cv2.imwrite(output_pgm2, output2)
    exercise_02b_compare(input_pgm2, output_pgm2, output_txt2)
    
    #display the output image
    cv2.imshow("Input Image1", image1)
    cv2.imshow("Output Image1", output1)
    cv2.imshow("compare_image1",cv2.imread(compare_pgm1, cv2.IMREAD_GRAYSCALE))
    cv2.imshow("Input Image2", image2)
    cv2.imshow("Output Image2", output2)
    cv2.imshow("compare_image2",cv2.imread(compare_pgm2, cv2.IMREAD_GRAYSCALE))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

