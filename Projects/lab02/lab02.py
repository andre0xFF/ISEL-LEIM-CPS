#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from lab01.lab01 import sawtooth_signal, quantify, uniform_midrise_quantizer


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

    mq, idx = quantify(m, vmax, vj, tj)

    bin = pcm_encode(idx, 3)
    dec = pcm_decode(bin)

    print('Quantize signal must be equal to (Encode > Decode > Dequantize): {}'. format(np.array_equal(mq, vj[dec])))


def exercise_01():
    signal = sawtooth_signal()
    vmax = np.max(np.abs(signal))
    r = 3

    delta_q = (2 * vmax) / (np.power(2, r))
    vj, tj = uniform_midrise_quantizer(vmax, delta_q)
    mq, idx = quantify(signal, vmax, vj, tj)

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
    # TODO: SNR calculation
    fs, m = wav.read("lab01/som_8_16_mono.wav")

    vmax = np.max(np.abs(m))
    r = np.array([3, 5, 8])

    for i in r:
        delta_q = (2 * vmax) / (np.power(2, i))
        vj, tj = uniform_midrise_quantizer(vmax, delta_q)
        mq, idx = quantify(m, vmax, vj, tj)

        filename = 'lab02/som_8_16_quantize_{}.wav'.format(i)
        wav.write(filename, fs, mq.astype('int16'))

        bin = pcm_encode(idx, i)
        dec = pcm_decode(bin)

        filename = 'lab02/som_8_16_quantize_encode_decode_dequantize_{}.wav'.format(i)
        wav.write(filename, fs, vj[dec].astype('int16'))


def exercise_03():
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

    G = np.hstack((np.identity(11), P))

    C = np.logical_xor(m, G)
    # or
    # C = np.dot(m, G) % 2


def exercise_04():
    pass


def exercise_05():
    pass


def exercise_06():
    pass


if __name__ == '__main__':
    main()
