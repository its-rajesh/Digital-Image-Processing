from matplotlib import pyplot as plt
import numpy as np
import cv2
import scipy
from scipy import signal
from math import log10, sqrt


'''
READING AN IMAGE
'''
image = cv2.imread('cameraman.jpeg', cv2.IMREAD_GRAYSCALE)
plt.imshow(image, cmap='gray')
plt.title('x(m, n): Input Image')
plt.show()


'''
CREATING GAUSSIAN KERNEL
Inputs: size of kernel, variance
Returns: gaussian kernel
'''
def gauss_kernal(size, sigma):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

noise = gauss_kernal(5, 4)
plt.imshow(noise)
plt.title('Noise levels')
plt.colorbar()
plt.show()

'''
CREATING MOTION BLUR KERENEL
Inputs: size of the kernel
Returns: motion blur kernel
'''

def motionblurKernel(size):
    motion_blur = np.zeros((size, size))
    motion_blur[int((size-1)/2), :] = np.ones(size)
    motion_blur = motion_blur / size
    return motion_blur

blur = motionblurKernel(10)
plt.imshow(blur)
plt.title('Motion Blur Kernel')
plt.colorbar()
plt.show()

'''
APPLYING MOTION BLUR TO IMAGE
Inputs: Image, size of kernel if required. by default it takes 3 for faster computations
Returns: Blurred Image, Applied kernel
'''
def apply_motion_blur(image, size=3):
    
    img = np.copy(image)
    kernel = np.eye(size)/size       
    img = scipy.signal.convolve2d(img, kernel, mode='valid')
    img = np.uint8(img)
    
    return img, kernel
    
'''
APPLYING GAUSSIAN BLUR TO IMAGE
Inputs: Image, size of kernel if required. by default it takes 3 for faster computations
Returns: Blurred Image, Applied kernel
'''
def apply_gaussian_blur(image, size=3):
    
    img = np.copy(image)
    kernel = gaussian(size, size/3).reshape(size, 1)
    kernel = np.dot(kernel, kernel.transpose())
    kernel /= np.sum(kernel)
    img = scipy.signal.convolve2d(img, kernel, mode='valid')
    img = np.uint8(img)
    
    return img, kernel
    
'''
APPLYING BOX BLUR TO IMAGE
Inputs: Image, size of kernel if required. by default it takes 3 for faster computations
Returns: Blurred Image, Applied kernel
''' 
def apply_box(image, size=3):
    
    img = np.copy(image)
    kernel = np.ones((size, size))/(size*size)   
    img = scipy.signal.convolve2d(img, kernel, mode='valid')
    img = np.uint8(img)
    
    return img, kernel
    
'''
APPLY ANY ONE BLUR FUNCTIONS.
Im applying motion blur, but works with all above blurs.
'''
Blurred_image, kernel = apply_motion_blur(image)
plt.imshow(Blurred_image, cmap='gray')
plt.title('Blurred Image')
plt.show()

'''
ADDING NOISE TO THE IMAGE
Inputs: Blurred Image, Variance of noise
Returns: Nosiy Blurred Image
'''
def add_gaussian_noise(img, variance):
    image = np.copy(img).astype(float)
    gaussian = np.random.normal(0, variance, np.shape(image))
    
    noisy_image = image + gaussian
    noisy_image = np.round(noisy_image)
    
    #Random values may exceed 0, 255: so keeping bounds
    noisy_image[np.where(image<0)] = 0
    noisy_image[np.where(image>255)] = 255
    
    noisy_image = np.uint8(noisy_image)
    
    return noisy_image
    
'''
Adding noise with variance 4.
'''  
noisy_blurred_image = add_gaussian_noise(Blurred_image, 4)
plt.imshow(noisy_blurred_image, cmap='gray')
plt.title('Noisy Blurred Image')
plt.show()

'''
WIENER FILTER IMPLEMENTATION
Inputs: Noisy Blurred Image, Applied PSF and SNR (k) 
Returns: Reconstructed Image
'''
def filters(blurred_noisy_image, h, k=5):
    
    image = np.copy(blurred_noisy_image)
    
    h = np.pad(h, [(0, image.shape[0] - h.shape[0]), (0, image.shape[1] - h.shape[1])], 'constant')
    
    dft_image = np.fft.fft2(image)
    dft_h = np.fft.fft2(h)
    
    dft_h_conjugate = np.conj(dft_h)
    
    weiner_filter = dft_h_conjugate/((np.abs(dft_h)*np.abs(dft_h))+(1/k))
    
    dft_output_image = dft_image * weiner_filter
    output_image = np.fft.ifft2(dft_output_image)
    
    output_image = np.uint8(np.abs(output_image))
    
    return output_image
    
    
reconst_image = filters(noisy_blurred_image, kernel)
plt.imshow(reconst_image, cmap='gray')
plt.title('Reconstructed Image')
plt.show()