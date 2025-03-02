# public libraries 
import sys
import os
import cv2
import numpy as np
# prviate libraries
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Exercises_13ab.exercise_13a_minimun import read_input_file, read_pgm_image, get_neighbors, compute_flat_zone, is_regional_minimum
from Exercises_13ab.exercise_13b_maxium import is_regional_maximum
from exercise_13c_minium import compute_regional_extrema
##

if __name__ == "__main__":
    image = read_pgm_image("Exercises_13cd/immed_gray_inv_20051218_frgr4.pgm")
    output = compute_regional_extrema(image, extrema_type="max")
    cv2.imwrite("Exercises_13cd/exercise_13d_output_01.pgm", output)

