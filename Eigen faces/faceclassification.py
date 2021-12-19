import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

folder = "/Users/rajeshr/Desktop/DIPLab4/faces"

training_images = []

for i in range(1, 41):
    path = folder+'/s'+str(i)
    #print(path)
    for j in range(1, 6):
        full_path = path+'/'+str(j)+'.jpg'
        #print(full_path)
        #training_images.append(cv2.imread(full_path, cv2.IMREAD_GRAYSCALE))
        training_images.append(full_path)

print(len(training_images))
for i in range(len(training_images)):
    #image = np.array(training_images[i], dtype=np.uint8)
    image = cv2.imread(training_images[i], cv2.IMREAD_GRAYSCALE)
    img = np.array(image, dtype=np.uint8)
    plt.subplot(15, 14, i+1)
    plt.imshow(img, cmap='gray')

plt.show()


def pil_grid(images, max_horiz=np.iinfo(int).max):
    n_images = len(images)
    n_horiz = min(n_images, max_horiz)
    h_sizes, v_sizes = [0] * n_horiz, [0] * (n_images // n_horiz)
    for i, im in enumerate(images):
        h, v = i % n_horiz, i // n_horiz
        h_sizes[h] = max(h_sizes[h], im.size[0])
        v_sizes[v] = max(v_sizes[v], im.size[1])
    h_sizes, v_sizes = np.cumsum([0] + h_sizes), np.cumsum([0] + v_sizes)
    im_grid = Image.new('RGB', (h_sizes[-1], v_sizes[-1]), color='white')
    for i, im in enumerate(images):
        im_grid.paste(im, (h_sizes[i % n_horiz], v_sizes[i // n_horiz]))
    return im_grid

pil_grid(training_images, 15)