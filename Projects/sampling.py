# https://www.gaussianwaves.com/2014/07/sampling-a-signal-in-matlab/

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

# linspace: Evenly spaced numbers over a specified interval (start, stop, number of points)
# arange: Evenly space values within a given interval (start, stop, step)

fs = 500e3
f = 10e3
nCyl = 5
t = np.arange(0, nCyl * 1 / f, 1 / fs)
x = np.cos(2 * np.pi * f * t)
plt.plot(t, x)

fs1 = 30e3
t1 = np.arange(0, nCyl * 1 / f, 1 / fs1)
x1 = np.cos(2 * np.pi * f * t1)
plt.stem(t1, x1)

plt.show()

x = np.linspace(0, 10, 20, endpoint=False)
y = np.cos(-x**2/6.0)
plt.plot(x, y)

f = scipy.signal.resample(y, 100)
xnew = np.linspace(0, 10, 100, endpoint=False)
plt.stem(xnew, f)

plt.show()
