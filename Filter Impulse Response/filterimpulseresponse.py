import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

#np.seterr(divide='ignore', invalid='ignore')

# Reading an Image
img = cv2.imread('/Users/rajeshr/Desktop/peppers.png', cv2.IMREAD_GRAYSCALE)

'''
______________________________________________________________________________________________
                        Impulse Response of an Averaging Filter
______________________________________________________________________________________________
'''
####################### CREATING KERNAL #########################

size = 25
Avg_kernel = np.ones((size,size),np.float32)/(size*size)
row, col = Avg_kernel.shape

####################### IMPULSE RESPONSE #########################

# 3D Plotting Essentials
fig = plt.figure(figsize=(6, 6))
ax = plt.axes(projection='3d')
x = np.linspace(0, row, col)
y = np.linspace(0, col, row)
X, Y = np.meshgrid(x, y)
Z = Avg_kernel

# Displaying the impulse response of Averaging Filter
ax.plot_surface(X, Y, Z, color='green')
ax.set_title('Averaging Filter Impulse Response')
plt.show()

####################### FREQUENCY RESPONSE #########################

Avg_kernel_fft = np.fft.fftshift(np.fft.fft2(Avg_kernel))
#Avg_kernel_fft = np.log(abs(Avg_kernel_fft))
row, col = Avg_kernel_fft.shape

# 3D Plotting Essentials
fig = plt.figure(figsize=(6, 6))
ax = plt.axes(projection='3d')
x = np.linspace(0, row, col)
y = np.linspace(0, col, row)
X, Y = np.meshgrid(x, y)
Z = Avg_kernel_fft

# Displaying the Frequency response of Averaging Filter
ax.plot_surface(X, Y, Z)
ax.set_title('Averaging Filter Frequency Response')
plt.show()

'''
______________________________________________________________________________________________
                        Impulse Response of an Gaussian Filter
______________________________________________________________________________________________
'''

####################### CREATING KERNAL #########################

# Creating Guassian Kernel Mask in 2D
def gkern(kernlen=21, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

####################### IMPULSE RESPONSE #########################

# Getting the 2D kernel and its shape
size, sigma = 25, 3 #Enter Size and sigma values
Guass_impulse = gkern(size, sigma)
row, col = Guass_impulse.shape

# 3D Plotting Essentials
fig = plt.figure(figsize=(6, 6))
ax = plt.axes(projection='3d')
x = np.linspace(0, row, col, endpoint=True)
y = np.linspace(0, col, row, endpoint=True)
X, Y = np.meshgrid(x, y)
Z = Guass_impulse

# Displaying the impulse response of Averaging Filter
ax.plot_surface(X, Y, Z)
#ax.plot_wireframe(X, Y, Z, color='blue')
ax.set_title('Gaussian Filter Impulse Response')
plt.show()

####################### FREQUENCY RESPONSE #########################

Guass_freq = np.fft.fft2(Guass_impulse)
Guass_freq = np.log(abs(Guass_freq))
row, col = Guass_freq.shape

# 3D Plotting Essentials
fig = plt.figure(figsize=(6, 6))
ax = plt.axes(projection='3d')
x = np.linspace(0, row, col, endpoint=True)
y = np.linspace(0, col, row, endpoint=True)
X, Y = np.meshgrid(x, y)
Z = Guass_freq

# Displaying the frequency response of Gaussian Filter
ax.plot_surface(X, Y, Z)
#ax.plot_wireframe(X, Y, Z, color='blue')
ax.set_title('Gaussian Filter Frequency Response')
plt.show()

'''
______________________________________________________________________________________________
                        Impulse Response of an Motion Blur
______________________________________________________________________________________________
'''
####################### CREATING KERNAL #########################

size = 25
motion_blur = np.zeros((size, size))
motion_blur[int((size-1)/2), :] = np.ones(size)
motion_blur = motion_blur / size
row, col = motion_blur.shape

####################### IMPULSE RESPONSE #########################

# 3D Plotting Essentials
fig = plt.figure(figsize=(6, 6))
ax = plt.axes(projection='3d')
x = np.linspace(0, row, col, endpoint=True)
y = np.linspace(0, col, row, endpoint=True)
X, Y = np.meshgrid(x, y)
Z = motion_blur

# Displaying the impulse response of Averaging Filter
ax.plot_surface(X, Y, Z)
#ax.plot_wireframe(X, Y, Z, color='blue')
ax.set_title('Motion Blur Impulse Response')
plt.show()

####################### FREQUENCY RESPONSE #########################

motion_freq = np.fft.fftshift(np.fft.fft2(motion_blur))
motion_freq = np.log(abs(motion_freq))
row, col = motion_freq.shape

# 3D Plotting Essentials
fig = plt.figure(figsize=(6, 6))
ax = plt.axes(projection='3d')
x = np.linspace(-row, row, col, endpoint=True)
y = np.linspace(-col, col, row, endpoint=True)
X, Y = np.meshgrid(x, y)
Z = motion_freq

# Displaying the frequency response of Gaussian Filter
ax.plot_surface(X, Y, Z)
#ax.plot_wireframe(X, Y, Z, color='blue')
ax.set_title('Motion Blur Frequency Response')
plt.show()
