"""this module is used to check 
the idempotance of the 'opening-closing' alternated filters for a particular case"""


# public libraries
import sys
import os
import cv2

# prviate libraries
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Exercises_06ab.exercise_06b_opening_closing import exercise_06b_opening_closing
from Exercises_02ab.exercise_02b_compare import exercise_02b_compare
##


def excercise_07b(i,input_file, output_file1, output_file2, method="numpy"):
    """Check the idempotance of the 'opening-closing' alternated filters"""

    # apply the ' opening-closing' alternated filters
    exercise_06b_opening_closing(i, input_file, output_file1, method)
    exercise_06b_opening_closing(i, output_file1, output_file2, method)

    # compare the two results
    flag = exercise_02b_compare(output_file1, output_file2)
    if flag:
        print("The 'opening-closing' alternated filters are idempotent")
    else:
        print("The 'opening-closing' alternated filters are not idempotent")
    
    return flag

# main() function
if __name__ == "__main__":
    # initialize the parameters
    i=1 # 3x3 结构元素
    #i=2 # 5x5 结构元素

    #read the input image
    input_file = "Exercises_07ab/cam_74.pgm"
    output_file1 = "Exercises_07ab/exercise_07b_output_01.pgm"
    output_file2 = "Exercises_07ab/exercise_07b_output_02.pgm"

    # run the test
    excercise_07b(i, input_file, output_file1, output_file2, method="list")

    # display the images
    img_original = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    img_filtered1 = cv2.imread(output_file1, cv2.IMREAD_GRAYSCALE)
    img_filtered2 = cv2.imread(output_file2, cv2.IMREAD_GRAYSCALE)

    if img_original is not None and img_filtered1 is not None and img_filtered2 is not None:
        cv2.imshow("Original Image", img_original)
        cv2.imshow(f"Opening-Closing ( 1st)", img_filtered1)
        cv2.imshow(f"Opening-Closing ( 2nd)", img_filtered2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error: Unable to load images for display.")
