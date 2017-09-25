#!/usr/bin/env python3
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt


def main():
    exercise_01()


def exercise_01():
    # a)
    f = 3014
    a = 20000
    fs = 8000
    length = 1

    # number of points along time
    t1 = np.linspace(0, length, fs * length)
    # t = np.arange(0, length, 1 / fs)

    # signal: x(n) = A * cos(2 * pi * f * (n / Fs))
    x1 = sinwave(a, f, t1)

    wav.write('sound_01.wav', fs, x1.astype('int16'))

    # b i)
    fs = 4000
    x2 = x1[0:len(x1):int(8000 / 4000)]
    t2 = t1[0:len(t1):int(8000 / 4000)]

    # or we could recalculate the signal
    # t2 = np.linspace(0, length, fs * length)
    # x2 = sinwave(a, f, t2)

    wav.write('sound_01_b_i.wav', fs, x2.astype('int16'))

    figure = plt.figure()

    ax1 = figure.add_subplot(121)
    ax1.axis([0, 0.01, -20000, 20000])
    ax1.plot(t1, x1)
    ax2 = figure.add_subplot(122)
    ax2.axis([0, 0.01, -20000, 20000])
    ax2.plot(t2, x2)

    # slicing can also work instead of modyfing the plot axis
    # plt.plot(t[0:20], x[0:20])

    plt.tight_layout()
    plt.show()

    # b ii)
    fs, x3 = wav.read("som_8_16_mono.wav")

    fs4 = 1000
    x4 = x3[0:len(x3):int(8000 / 1000)]

    figure = plt.figure()
    ax1 = figure.add_subplot(121)
    ax1.plot(x3)
    ax2 = figure.add_subplot(122)
    ax2.plot(x4)

    plt.show()

    wav.write('sound_01_b_ii.wav', fs4, x3.astype('int16'))


def sinwave(a, f, t):
    return a * np.cos(2 * np.pi * f * t)


# r: number of bits per sample
# vmax: max value to quantify
# type-: quantifier-type (midrise or midtread)
def quantify(r, vmax, type):
    interval = (2 * vmax) / (np.power(2, r))

    if type == "midtread":
        pass

    if type == "midraise":
        pass

    vj = np.arange(0, vmax * 2 - interval, interval)
    # tj
    pass

if __name__ == "__main__":
    main()
