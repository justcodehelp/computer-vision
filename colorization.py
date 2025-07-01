import cv2
import numpy as np

gray_image = cv2.imread("R:\MSc IT\sem 2\computer vision\grayimg.jpeg", cv2.IMREAD_GRAYSCALE)

color_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

color_lookup_table = np.zeros((256, 1, 3), dtype=np.uint8)
for i in range(256):
    color_lookup_table[i, 0, 0] = i  
    color_lookup_table[i, 0, 1] = 127  
    color_lookup_table[i, 0, 2] = 255 - i  

colorized_image = cv2.LUT(color_image, color_lookup_table)

cv2.imshow('Grayscale Image', gray_image)
cv2.imshow('Colorized Image', colorized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
