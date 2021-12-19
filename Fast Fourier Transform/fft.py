import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2gray


image1 = imread('/Users/rajeshr/Desktop/7.gif')
image1_grey = rgb2gray(image1)
plt.figure(num=None, figsize=(8, 6), dpi=80)
plt.imshow(image1_grey, cmap='gray')
plt.show()

image1_fourier = np.fft.fftshift(np.fft.fft2(image1_grey))
plt.figure(num=None, figsize=(8, 6), dpi=80)
plt.imshow(np.log(abs(image1_fourier)), cmap='gray')
plt.title("FFT With shift")
plt.show()

image1_fftshift = np.fft.fft2(image1_grey)
plt.figure(num=None, figsize=(8, 6), dpi=80)
plt.imshow(np.log(abs(image1_fftshift)), cmap='gray')
plt.title("FFT Without shift")
plt.show()


image2 = imread('/Users/rajeshr/Desktop/33.gif')
image2_grey = rgb2gray(image2)
plt.figure(num=None, figsize=(8, 6), dpi=80)
plt.imshow(image2_grey, cmap='gray')
plt.show()

image2_fft = np.fft.fft2(image2_grey)
plt.figure(num=None, figsize=(8, 6), dpi=80)
plt.imshow(np.log(abs(image2_fft)), cmap='gray')
plt.title("FFT Without shift")
plt.show()

image2_fourier = np.fft.fftshift(np.fft.fft2(image2_grey))
plt.figure(num=None, figsize=(8, 6), dpi=80)
plt.imshow(np.log(abs(image2_fourier)), cmap='gray')
plt.title("FFT With shift")
plt.show()
