import pdb
import numpy as np
import scipy.fftpack
import matplotlib.pyplot as plt
import numpy as np
import math

arr = np.zeros(2000)
length_per_pixel = 1  # mm
f = 0.01
amp = 3
amp2 = 1
f2 = 0.06
for x in range(len(arr)):
    i = 0
    true_position = x * length_per_pixel
    val = 10 + amp * np.sin(true_position * 2 * math.pi * f) + \
        amp2 * np.sin(true_position * 2 * math.pi * f2)
    arr[x] = val
    pass


fig, ax = plt.subplots(3, 1)
ax[0].plot(arr)


total_t = len(arr)
fs = 1
N = int(total_t * fs)
ts = np.linspace(0, total_t, N)
y = arr
yf = scipy.fftpack.fft(y)
xf = np.linspace(0, fs / 2, N // 2)
fftfreqs = scipy.fftpack.fftfreq(N)
fftvals = (2 / N) * np.abs(yf[:int(N // 2)])
r,  = np.where(fftfreqs == f)
ax[0].set_title("Signal at %s Hz" % f)
ax[1].plot(xf, fftvals)

assert fftvals[r] == amp

# filter signal for the f2 freq
rss, = np.where(np.abs(np.fft.fftshift(fftfreqs)) > 0.04)  # cut off f2 signal
yfs = np.fft.fftshift(yf)
yfs[rss] = 0

# invert filtered signal
outval = np.fft.ifft(yfs)
ax[2].set_title("cleaned signal")
ax[2].plot(np.abs(outval))
fig.show()
