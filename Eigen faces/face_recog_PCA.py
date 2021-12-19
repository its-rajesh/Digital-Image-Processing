
#
#		READING FOLDER OF IMAGE
#

import cv2
import os
from matplotlib import pyplot as plt


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images
    

inputs = [] # Contains all .jpg and .pgm
folder = '/Users/rajeshr/Desktop/DIPLab4/faces/s1'
inputs = load_images_from_folder(folder)
print(len(inputs))


training_input = [] #Contains only first 5 images .jpg
for i in range(0, int(len(inputs)/4)):
	training_input.append(inputs[i*2])

print(len(training_input))
row, col, channel = training_input[0].shape
print(row, col)

#
#		STITCHING OF IMAGES INTO SINGLE
#
from PIL import Image

width = col*5
height = row
print(width, height)

new_im = Image.new('RGB', (width, height))

new_im.save('/Users/rajeshr/Desktop/DIPLab4/faces/test.jpg')


'''
import sys
from PIL import Image

images = [Image.open(x) for x in ['Test1.jpg', 'Test2.jpg', 'Test3.jpg']]
widths, heights = zip(*(i.size for i in images))

total_width = sum(widths)
max_height = max(heights)

new_im = Image.new('RGB', (total_width, max_height))

x_offset = 0
for im in images:
  new_im.paste(im, (x_offset,0))
  x_offset += im.size[0]

new_im.save('test.jpg')

'''

#
#

import numpy as np

train = []
for i in range(1, 41):
	for j in range(1, 6):
		im = cv2.imread(folder+str(i)+'/'+str(j)+'.jpg', 0)
		if im.shape != None:
			print(im.shape)
			im_num = np.reshape(im, (im.shape[0]*im.shape[1], 1))
			train.append(im_num)
		else:
			pass
		
		
mean = np.array(im_num).mean(axis=0)
plt.imshow(np.reshape(mean, (im.shape[0], im.shape[1])), cmap='gray')

