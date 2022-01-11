import numpy as np
import scipy.fftpack
import matplotlib.pyplot as plt
import numpy as np
import math

arr = np.zeros(2000)
length_per_pixel = 1  # mm
f = 0.01
amp = 3
for x in range(len(arr)):
    i = 0
    true_position = x * length_per_pixel
    val = 10 + amp * np.sin(true_position * 2 * math.pi * f)
    arr[x] = val
    pass


fig, ax = plt.subplots(2, 1)
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

fig.show()
assert fftvals[r] == amp
