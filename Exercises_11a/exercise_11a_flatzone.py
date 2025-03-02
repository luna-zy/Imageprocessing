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

    input_txt= "Exercises_11a/exercise_11a_input_01.txt"
    input_pgm= "Exercises_11a/gran01_64.pgm"
    output_pgm="Exercises_11a/exercise_11a_outpu_01.pgm"
    

    x, y, connectivity, LABEL_FZ = read_input_file(input_txt)
    image = cv2.imread(input_pgm, cv2.IMREAD_GRAYSCALE)
    output = compute_flat_zone(image, x, y, connectivity, LABEL_FZ)
    cv2.imwrite(output_pgm, output)
    
    #display the output image
    cv2.imshow("Input Image", image)
    cv2.imshow("Output Image", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

