#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from lab01.lab01 import sawtooth_signal, quantize, uniform_midrise_quantizer, quantization_interval, snr_theoric


def main():
    example()
    exercise_01()
    exercise_02()
    exercise_03()
    exercise_04()
    exercise_05()
    exercise_06()


def example():
    # sample 3, page 85, midrise
    vmax = 1
    delta_q = 2 * vmax / 8
    vj, tj = uniform_midrise_quantizer(vmax, delta_q)

    n = np.arange(0, 8)
    m = np.round(np.sin(2 * np.pi * (np.float(1300) / 8000) * n), decimals=3)

    mq, idx = quantize(m, vmax, vj, tj)

    bin = pcm_encode(idx, 3)
    dec = pcm_decode(bin)

    print('Quantized signal must be equal to (Encode > Decode > Dequantize): {}'. format(np.array_equal(mq, vj[dec])))


def exercise_01():
    signal = sawtooth_signal()
    vmax = np.max(np.abs(signal))
    r = 3

    delta_q = quantization_interval(vmax, r)
    vj, tj = uniform_midrise_quantizer(vmax, delta_q)
    mq, idx = quantize(signal, vmax, vj, tj)

    bin = pcm_encode(idx, r)
    dec = pcm_decode(bin)


def pcm_encode(idx, r):
    # New data type to divide an int32 variable into 4 int8 variables
    dt = np.dtype((np.int32, {'f0': (np.uint8, 3), 'f1': (np.uint8, 2), 'f2': (np.uint8, 1), 'f3': (np.uint8, 0)}))

    # Convert the vector into new data type
    idx_uint8 = idx.view(dtype=dt)

    # Pack an numpy array with the 4 uint8 variables
    idx_uint8 = np.array([idx_uint8['f0'], idx_uint8['f1'], idx_uint8['f2'], idx_uint8['f3']])

    # Transpose so we get each number by row
    idx_uint8 = np.transpose(idx_uint8)

    # Convert to binary
    idx_bin = np.unpackbits(idx_uint8, axis=1)

    # Slice into the desired number of bits
    idx_bin = idx_bin[:, len(idx_bin[0]) - r:len(idx_bin[0])]

    return idx_bin


def pcm_decode(bits):
    return bits.dot(1 << np.arange(bits.shape[-1] - 1, -1, -1))


def gray_encode(idx, r):
    pass


def gray_decode(bits):
    pass


def exercise_02():
    fs, m = wav.read("lab01/som_8_16_mono.wav")

    vmax = np.max(np.abs(m))
    r = np.array([3, 5, 8])
    snr = np.arange(len(r), dtype='float')

    for i in range(len(r)):
        delta_q = quantization_interval(vmax, r[i])
        vj, tj = uniform_midrise_quantizer(vmax, delta_q)
        mq, idx = quantize(m, vmax, vj, tj)

        filename = 'lab02/som_8_16_quantize_{}bits.wav'.format(r[i])
        wav.write(filename, fs, mq.astype('int16'))

        bin = pcm_encode(idx, r[i])
        dec = pcm_decode(bin)

        filename = 'lab02/som_8_16_quantize_encode_decode_dequantize_{}bits.wav'.format(r[i])
        wav.write(filename, fs, dequantize(vj, dec).astype('int16'))

        p = np.sum(m * m) / len(m)
        snr[i] = snr_theoric(r[i], p, vmax)


def dequantize(vj, indexes):
    return vj[indexes]


def exercise_03():
    n = 15
    r = 11

    m = sawtooth_signal()

    vmax = np.max(np.abs(m))
    delta_q = quantization_interval(vmax, r)
    vj, tj = uniform_midrise_quantizer(vmax, delta_q)
    mq, idx = quantize(m, vmax, vj, tj)

    # Parity matrix
    P = np.array([
        [1, 1, 1, 1],
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 0, 0, 1],
        [1, 1, 0, 0],
    ])

    # Encode and codify with Hamming(15, 11)
    bin = pcm_encode(idx, r)
    c = hamming(bin, P, n, r)

    # Generate an error to simulate the channel
    error = np.random.binomial(1, 0.01, len(c))
    y = (c + error) % 2


def hamming(message, P, n, r):
    G = np.hstack((np.identity(15 - len(P[0])), P))

    # C = np.logical_xor(m, G)
    C = np.dot(message, G) % 2
    return np.ndarray.flatten(C)


def error_correction(message, P):
    H = np.vstack((P, np.identity(len(P[0]))))
    S = np.dot(message, H)

    i = np.nonzero(S)[0]

    if len(i) == 0:
        return S


def exercise_04():
    pass


def exercise_05():
    pass


def exercise_06():
    pass


if __name__ == '__main__':
    main()
