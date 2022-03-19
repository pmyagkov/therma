import matplotlib.image as img
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

# image = img.imread('./thermal-image.jpeg')

""" for i in range(0, image.shape[0]):
  for j in range(0, image.shape[1]):
    image[i][j] = [image[i][j][0], 0, 0] """


# img = Image.fromarray(image, 'RGB')


  # np.apply_along_axis(lambda color: color[0], i, image)

# image = cv2.imread('./thermal-image.jpeg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#iamge = cv2.threshold(iamge, 0, 255,
	#cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# img = Image.open('./thermal-image.jpeg')
# image = cv2.imread('./thermal-image.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imwrite('./img/1.jpg', image)
image2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
image3 = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
image4 = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

cv2.imwrite('./img/1.jpg', image)
cv2.imwrite('./img/2.jpg', image2)
cv2.imwrite('./img/3.jpg', image3)
cv2.imwrite('./img/4.jpg', image4)

""" cv2.imshow('image', image2)
cv2.imshow('image', image3)
cv2.imshow('image', image4)
cv2.waitKey(0) """
# img = Image.fromarray(image, 'RGB')
# img.show()

# plt.matshow(image)
# plt.show()
