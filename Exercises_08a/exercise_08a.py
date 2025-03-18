"""Exercise 08a. Let I be the input image in file isn_256.pgm, 
which has abinary impulsive noise added ("salt−and−pepper" noise). 
Let B be astructuring element square of size 3x3"""
"""Compute: Filter 1: opening_B (I)
            Filter 2: closing_B (I)
            Filter 3: closing_B (opening_B (I))
            Filter 4: opening_B (closing_B (I))
Indicate which are the two best filters to eliminate 
the noise in output file: 'exercise_08a_output_01.txt' 
(particularly,the first two lines should contain the filter numbers (1-4))."""

"""iuput_image=Exercises_08a/isn_256.pgm"""

# public libraries 
import sys
import os
import cv2
import numpy as np
# prviate libraries
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Exercises_06ab.exercise_06a_closing_opening import exercise_06a_closing_opening
from Exercises_02ab.exercise_02b_compare import exercise_02b_compare
##

def exercise_08a(i, input_file, output_file1, output_file2, output_file3, output_file4, method="numpy"):
    """Check the idempotance of the 'closing-opening' alternated filters for a particular case"""

    # apply the 'closing-opening' alternated filters
    exercise_06a_closing_opening(i, input_file, output_file1, method)
    exercise_06a_closing_opening(i, output_file1, output_file2, method)
    exercise_06a_closing_opening(i, output_file2, output_file3, method)
    exercise_06a_closing_opening(i, output_file3, output_file4, method)

    # compare the two results
    flag1 = exercise_02b_compare(output_file1, output_file2) 
    flag2 = exercise_02b_compare(output_file3, output_file4)
    if flag1:
        print("The 'closing-opening' alternated filters are idempotent")
    else:
        print("The 'closing-opening' alternated filters are not idempotent")
    
    return flag1, flag2

# main() function

if __name__ == "__main__":
    
    inputfile = "Exercises_08a/isn_256.pgm"
    output_file1 = "Exercises_08a/exercise_08a_output_01.pgm" 
    output_file2 = "Exercises_08a/exercise_08a_output_02.pgm"
    output_file3 = "Exercises_08a/exercise_08a_output_03.pgm"
    output_file4 = "Exercises_08a/exercise_08a_output_04.pgm"
    output_txt="Exercises_08a/exercise_08a_output_01.txt"

    exercise_08a(1, inputfile, output_file1, output_file2, output_file3, output_file4, method="numpy")
    with open(output_txt, "w") as f:
        f.write("1\n2")
    # display the images
    img_original = cv2.imread(inputfile, cv2.IMREAD_GRAYSCALE)
    img_filtered1 = cv2.imread(output_file1, cv2.IMREAD_GRAYSCALE) 
    img_filtered2 = cv2.imread(output_file2, cv2.IMREAD_GRAYSCALE)
    img_filtered3 = cv2.imread(output_file3, cv2.IMREAD_GRAYSCALE)
    img_filtered4 = cv2.imread(output_file4, cv2.IMREAD_GRAYSCALE)

    if img_original is not None and img_filtered1 is not None and img_filtered2 is not None and img_filtered3 is not None and img_filtered4 is not None:
        cv2.imshow("Original Image", img_original)
        cv2.imshow(f"Opening (1st)", img_filtered1)
        cv2.imshow(f"Closing (2nd)", img_filtered2)
        cv2.imshow(f"Closing-Opening (3rd)", img_filtered3)
        cv2.imshow(f"Opening-Closing (4th)", img_filtered4)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error: One or more images are missing")


