#!/usr/bin/env python3
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt


def main():
    exercise_04()


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


# exercise 2
# r: number of bits per sample
# vmax: max value to quantify
# type-: quantifier-type (midrise or midtread)
# vj: values of quantization
# tj: values of decision
def quantification_arrays(type, vmax, r):
    interval = (2 * vmax) / (np.power(2, r))

    if type == "midtread":
        vj = np.arange(-1 * vmax + interval, vmax + interval / 2, interval)
        tj = np.arange(-1 * vmax + interval * (3/2), vmax, interval)

    if type == "midrise":
        vj = np.arange(-1 * vmax + interval / 2, vmax, interval)
        tj = np.arange(-1 * vmax + interval, vmax, interval)

    return (vj, tj)


# exercise 3
def quantify(signal, type, vmax, r):
    vj, tj = quantification_arrays(type, vmax, r)

    xq = np.ones(len(vj)) * np.max(vj)
    # Insert vmax in tj at the end of list in order for vj and tj to have same size(8)
    tj = np.insert(tj, tj.size, vmax)

    for point in signal:
        eval = point <= tj

        if np.any(eval):
            print("")
            xq_value = vj[eval][0]
            xq = np.append(xq, xq_value)

    return xq


# Sawtooth signal
def sawtooth_signal():
    x = np.linspace(-20, 20, 1000)
    y = np.hstack((x, x, x, x, x))

    # plt.plot(y)
    # plt.show()
    # plt.close("all")

    return y


def exercise_04():
    signal = sawtooth_signal()
    vmax = np.max(np.abs(signal))

    # plt.plot(y)
    # plt.show()
    # plt.close("all")

    r = quantify(signal, 'midrise', vmax, 3)
    print(r)


    plt.plot(signal)
    plt.plot(r)
    plt.show()
    # hist, bins = np.histogram(r)
    plt.hist(r)
    # print(hist)
    # print(bins)
    plt.title("Histogram")
    plt.show()

    plt.close("all")

    print("")


if __name__ == "__main__":
    main()
