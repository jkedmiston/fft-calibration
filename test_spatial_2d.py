import numpy as np
import scipy.fftpack
import matplotlib.pyplot as plt
import numpy as np
import math

arr = np.zeros((400, 600))
length_per_pixel = 1  # mm
k1 = 0.05
k2 = 0.03
amp = 3
for x in range(arr.shape[1]):
    for y in range(arr.shape[0]):
        i = 0
        true_position = y * length_per_pixel, x * length_per_pixel
        # euler formula on
        # 10 + amp * e^ i * dot(k, x(t))
        val = 10 + amp * (np.cos(true_position[1] * 2 * math.pi * k1) *
                          np.cos(true_position[0] * 2 * math.pi * k2) -
                          np.sin(true_position[1] * 2 * math.pi * k1) *
                          np.sin(true_position[0] * 2 * math.pi * k2))
        arr[y, x] = val
        pass
    pass

fig, ax = plt.subplots(2, 1)
ax[0].imshow(arr)

N = arr.shape[0]
ftimage = np.fft.fft2(arr)
ftimage = np.fft.fftshift(ftimage)
# half = len(ftimage) // 2
# vals = (2/half) * np.abs(ftimage[:half, :half])
# https://scipython.com/book/chapter-6-numpy/examples/blurring-an-image-with-a-two-dimensional-fft/

rr = (1 / arr.shape[0]) * (2 / arr.shape[1]) * np.abs(ftimage)
# rr[arr.shape[0]//2, arr.shape[1]//2] /= 2
r, c = np.where(rr >= amp)
print(rr[r, c])
ax[1].imshow(rr)

rfreq = np.fft.fftshift(np.fft.fftfreq(arr.shape[0]))
val = rfreq[r]
np.testing.assert_array_equal(val, np.array([-k2, 0, k2]))
cfreq = np.fft.fftshift(np.fft.fftfreq(arr.shape[1]))
cval = cfreq[c]
np.testing.assert_array_equal(cval, np.array([-k1, 0, k1]))

fig.show()
