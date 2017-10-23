#!/usr/bin/env python3
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import scipy.signal as ss


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

    return vj, tj


# exercise 3
def quantify(signal, type, vmax, r):
    vj, tj = quantification_arrays(type, vmax, r)

    # Multiply by the max possible value in case the point is outside from tj limit
    # mq = np.ones(len(vj)) * np.max(vj)
    mq = np.array([])

    # Save the indexes of vj evaluation
    # Initialize as a matrix so can we can use np.unpackbits
    # and assign a first value so we can append later
    indexes = np.array([[254]], dtype='uint8')

    # Insert vmax in tj at the end of list in order for vj and tj to have same size
    tj = np.insert(tj, tj.size, vmax)

    # Loop every point in the signal and check if the point is lower or equal
    # than any tj elements (decision values)
    for point in signal:
        eval = point <= tj

        # Test whether any array element along a given axis evaluates to True
        if np.any(eval):
            xq_value = vj[eval][0]
            mq = np.append(mq, xq_value)

            # Get the index
            idx = np.nonzero(eval)[0][0]
            indexes = np.append(indexes, np.array([[idx]], dtype='uint8'), axis=0)

    # Remove the 1st fictitious value
    indexes = indexes[1:]

    return mq, indexes


# sawtooth signal
def sawtooth_signal():
    x = np.linspace(-20, 20, 1000)
    y = np.hstack((x, x, x, x, x))

    # plt.plot(y)
    # plt.show()
    # plt.close("all")

    return y


# signal = m(n)
# mq: signal quantified = eq(n) + mp(n)
def exercise_04():
    # a)
    signal = sawtooth_signal()
    vmax = np.max(np.abs(signal))
    r = 3

    mq, idx = quantify(signal, 'midrise', vmax, r)
    print("m(q) = {}".format(mq))

    plt.plot(signal)
    plt.plot(mq)
    # plt.plot(xQ)
    plt.show()

    # b)
    # The mp(n) starts with the same element as m(n)
    mp = np.copy(mq)
    mp = np.insert(mp, 0, signal[0])
    mp = np.delete(mp, len(mp) - 1)

    # error formula
    error = signal - mp

    # quantification of error
    error_quantified, idx = quantify(error, 'midrise', vmax, 3)

    plt.hist(error_quantified)
    plt.title("Histogram")
    plt.show()

    # c)
    px = np.sum(signal * signal) / len(signal)
    r = np.arange(3, 9)

    snr_t = np.arange(len(r), dtype='float')
    snr_p = np.arange(len(r), dtype='float')

    for i in range(len(r)):
        mq, idx = quantify(signal, 'midrise', vmax, r[i])
        eq = signal - mq
        pq = np.sum(eq * eq) / len(eq)
        snr_t[i] = 6 * r[i] + 10 * np.log10(3 * px / np.power(vmax, 2))
        snr_p[i] = 10 * np.log10(px / pq)

    # TODO: Graph


def exercise_05():
    # a)
    fs, x = wav.read("som_8_16_mono.wav")

    plt.hist(x)
    plt.title("Histogram")
    plt.show()

    # b)
    # TODO

    # c)
    vmax = np.max(np.abs(x))
    px = np.sum(x, x) / len(x)
    r = np.arange(3, 9)

    snr_t = np.arange(len(r), dtype='float')
    snr_p = np.arange(len(r), dtype='float')

    for i in range(len(r)):
        mq = quantify(x, 'midrise', vmax, r[i])
        eq = x - mq
        pq = np.sum(eq * eq) / len(eq)
        snr_t[i] = 6 * r[i] + 10 * np.log10(3 * px / np.power(vmax, 2))
        snr_p[i] = 10 * np.log10(px / pq)

    # TODO: Graphs and comments


def exercise_06():
    fs = 44000
    t = np.arange(fs)
    signal = 4 * np.cos(2 * np.pi * 10000 * t) + 10 * np.cos(2 * np.pi * 25000 * t)

    # a)
    fc1 = 20000
    low_pass_filter = ss.firwin(fc1, 1 / fc1, scale=True)
    x = np.fft.fft(signal)
    x_freq = np.fft.fftfreq(len(signal))

    plt.plot(x_freq, np.abs(x))
    plt.show()

    plt.plot(signal)
    plt.show()

    a = np.convolve(low_pass_filter, x)
    a_freq = np.fft.fftfreq(len(a)) * fs

    plt.plot(a_freq, np.abs(a))
    plt.show()

    # b)


if __name__ == "__main__":
    main()
