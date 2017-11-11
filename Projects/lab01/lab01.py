#!/usr/bin/env python3
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import scipy.signal as ss


def main():
    example()
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


def example():
    # sample 1, page 101
    vmax = 10
    r = 40 / (2 * 10)
    delta_q = quantization_interval(vmax, r)
    vj, tj = uniform_midtread_quantizer(vmax, delta_q)

    # sample 2, page 101
    vmax = 1
    r = 20 / (2 * 5)
    delta_q = quantization_interval(vmax, r)
    vj, tj = uniform_midrise_quantizer(vmax, delta_q)

    # sample 3, page 85, midrise
    vmax = 1
    delta_q = 2 * vmax / 8
    vj, tj = uniform_midrise_quantizer(vmax, delta_q)

    n = np.arange(0, 8)
    m = np.round(np.sin(2 * np.pi * (np.float(1300) / 8000) * n), decimals=3)
    plt.plot(m)

    mq, idx = quantize(m, vmax, vj, tj)
    plt.plot(mq)

    # sample 3, page 85, midtread
    vj, tj = uniform_midtread_quantizer(vmax, delta_q)
    mq, idx = quantize(m, vmax, vj, tj)
    plt.plot(mq)

    plt.show()


# exercise 2
# r: number of bits per sample
# vmax: max value to quantify
# type-: quantifier-type (midrise or midtread)
# vj: values of quantization
# tj: values of decision
def uniform_midtread_quantizer(vmax, delta_q):
    vj = np.arange(-1 * vmax + delta_q, vmax + delta_q / 2, delta_q)
    tj = np.arange(-1 * vmax + delta_q * (3 / 2), vmax, delta_q)

    return vj, tj


def uniform_midrise_quantizer(vmax, delta_q):
    vj = np.arange(-1 * vmax + delta_q / 2, vmax, delta_q)
    tj = np.arange(-1 * vmax + delta_q, vmax, delta_q)

    return vj, tj


# exercise 3
# mq: quantified signal
# idx: indexes of each quantified value
def quantize(signal, vmax, vj, tj):
    tj = np.insert(tj, len(tj), vmax)

    # Majorate mq array as default value
    mq = np.ones(len(signal)) * np.max(vj)

    # Majorate the index as default value
    # idx = np.ones(shape=(len(signal), 1), dtype='uint8') * len(vj)
    idx = np.ones(len(signal), dtype='uint32') * len(vj)

    # Loop every point in the signal and check if the point is lower or equal
    # than any tj elements (decision values)
    for i in range(len(signal)):
        eval = signal[i] <= tj

        # Test whether any array element along a given axis evaluates to True
        if np.any(eval):
            xq_value = vj[eval][0]
            mq[i] = xq_value

            # Get the index
            k = np.nonzero(eval)[0][0]
            idx[i] = k

    return mq, idx


# sawtooth signal
def sawtooth_signal():
    x = np.linspace(-20, 20, 1000)
    y = np.hstack((x, x, x, x, x))

    # plt.plot(y)
    # plt.show()
    # plt.close("all")

    return y


def vmax_calculation(signal):
    return np.max(np.abs(signal))


# signal = m(n)
# mq: signal quantified = eq(n) + mp(n)
def exercise_04():
    # a)
    signal = sawtooth_signal()
    vmax = vmax_calculation(signal)
    r = 3

    delta_q = quantization_interval(vmax, r)
    vj, tj = uniform_midrise_quantizer(vmax, delta_q)
    mq, idx = quantize(signal, vmax, vj, tj)

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

    # quantization of error
    error_quantified, idx = quantize(error, vmax, vj, tj)

    plt.hist(error_quantified)
    plt.title("Histogram")
    plt.show()

    # c)
    px = signal_power(signal)
    r = np.arange(3, 9)

    snr_t = np.arange(len(r), dtype='float')
    snr_p = np.arange(len(r), dtype='float')

    for i in range(len(r)):
        delta_q = quantization_interval(vmax, r[i])
        vj, tj = uniform_midtread_quantizer(vmax, delta_q)
        mq, idx = quantize(signal, vmax, vj, tj)

        eq = signal - mq
        pq = signal_power(eq)

        snr_t[i] = snr_theoric(r[i], px, vmax)
        snr_p[i] = snr_pratic(px, pq)

    # TODO: Graph


def signal_power(m):
    return np.sum(np.power(m, 2) / len(m))


def quantization_interval(v, r):
    return (2 * v) / (np.power(2, r))


def snr_theoric(r, p, v):
    return 6.02 * r + 10 * np.log10(3 * p / np.power(v, 2))


def snr_pratic(p_signal, p_quantized):
    return 10 * np.log10(p_signal / p_quantized)


def exercise_05():
    # a)
    fs, x = wav.read("som_8_16_mono.wav")

    plt.hist(x)
    plt.title("Histogram")
    plt.show()

    # b)
    # TODO

    # c)
    vmax = vmax_calculation(x)
    px = np.sum(x, x) / len(x)
    r = np.arange(3, 9)

    snr_t = np.arange(len(r), dtype='float')
    snr_p = np.arange(len(r), dtype='float')

    for i in range(len(r)):
        delta_q = quantization_interval(vmax, r[i])
        vj, tj = uniform_midtread_quantizer(vmax, delta_q)
        mq = quantize(x, vmax, vj, tj)

        eq = x - mq
        pq = np.sum(eq * eq) / len(eq)

        snr_t = snr_theoric(r[i], px, vmax)
        snr_p = snr_pratic(px, pq)

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
