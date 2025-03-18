import cv2 as cv
import numpy as np
from queue import Queue

type Image = cv.typing.MatLike

###############
# From the previous exercises
###############
def neighbors(x: int, y: int, image_x: int, image_y: int):
  offsets = [(-1, -1),(0, -1),(1, -1),
              (-1,  0),        (1,  0),
              (-1,  1),(0,  1),(1,  1)]

  for offset in offsets:
    neigh = (x+offset[0], y+offset[1])
    # make sure the pixel values are valid (not negative or outside of image bounds)
    if min(neigh) != -1 and neigh[0] < image_x and neigh[1] < image_y:
      yield neigh

def flatzones_number(img: Image):
  rows = img.shape[0]
  cols =  img.shape[1]
  labels = np.zeros((rows, cols, 1))

  current_label = 0
  fifo = Queue() # could also be a stack

  for px in range(cols):
    for py in range(rows):
      # if the pixel p is not labeled, we proceed with the
      # next label and start labeling it
      if labels[py, px] == 0:
        current_label += 1
        labels[py, px] = current_label

        # iterate through the neighbors of pixel p
        for pneigh_x, pneigh_y in neighbors(px, py, cols, rows):
          # if the neighboring pixel has the same intensity as p, add them
          # the the queue (for further exploration) and label it
          if img[pneigh_y, pneigh_x] == img[py, px]:
            fifo.put((pneigh_x, pneigh_y))
            labels[pneigh_y, pneigh_x] = current_label
        
        # expand the exploration for the flatzone based on the initial pixels added before
        while not fifo.empty():
          qx, qy = fifo.get()
          for qneigh_x, qneigh_y in neighbors(qx, qy, cols, rows):
            if img[qneigh_y, qneigh_x] == img[py, px] and labels[qneigh_y, qneigh_x] == 0:
              fifo.put((qneigh_x, qneigh_y)) # keep adding neighboring pixels to explore
              labels[qneigh_y, qneigh_x] = current_label

  return current_label



###############
# Wheel teeth
###############

def get_kernel(size: int):
  l = 2*size+1
  return np.ones((l, l), np.uint8)

def get_teeth(img: Image):
  _, img = cv.threshold(img, 23, 255, cv.THRESH_BINARY) # threshold image to binary

  # erode image
  erodedImg = cv.dilate(img, get_kernel(1))
  erodedImg = cv.erode(erodedImg, get_kernel(5))

  # get teeth
  teeth = img - erodedImg

  # get the outer ring of the wheel
  contours, _ = cv.findContours(teeth, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
  
  # put the teeth in a clean black background
  teeth.fill(0)
  cv.drawContours(teeth, contours, -1, 255, 1)

  # clean the teeth
  teeth = cv.dilate(teeth, get_kernel(2))
  teeth = cv.erode(teeth, get_kernel(2))

  # remove the teeth connections
  otherimg = cv.dilate(img, get_kernel(3))
  otherimg = cv.erode(otherimg, get_kernel(4))
  otherimg = cv.morphologyEx(otherimg, cv.MORPH_OPEN, get_kernel(10))

  teeth = cv.subtract(teeth, otherimg) 

  # teeth and teeth count
  return teeth, flatzones_number(teeth)

# python wheel_teeth.py
def main():
  input_img = cv.imread("TestImages\TestImages\wheel.png", cv.IMREAD_GRAYSCALE)

  teeth, teeth_count = get_teeth(input_img)

  cv.imwrite("wheel_teeth_out.png", teeth)
  print(f"Number of teeth: {teeth_count}")

main()