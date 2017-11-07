import numpy as np
import lab01
from lab01.lab01 import sawtooth_signal, quantify


def main():
    exercise_01()
    exercise_02()
    exercise_03()
    exercise_04()
    exercise_05()
    exercise_06()


def exercise_01():
    signal = sawtooth_signal()
    vmax = np.max(np.abs(signal))
    r = 3

    mq, idx = quantify(signal, 'midrise', vmax, r)

    bits = pcm_encode(idx, r)
    test = pcm_decode(bits)

    print('')


def pcm_encode(idx, r):
    idx = np.copy(idx)
    # bits = np.zeros(shape=(len(idx), r))

    bits = np.unpackbits(idx)



    # Codify quantification indexes
    bits = np.unpackbits(idx, axis=1)

    # Slice the codification to r bits
    return bits[:, len(bits[0]) - r:]


def pcm_decode(bits):
    # Check if bits is a bi-dimensional array
    if len(bits.shape) == 1:
        return -1

    # If the representation is already in 8 bits
    if len(bits[0]) == 8:
        return np.packbits(bits, axis=1)

    # Create an aux array with an 8 bit representation to work with .packbits()
    a = np.zeros(shape=(len(bits), 8), dtype='uint8')
    a[:, 8 - len(bits[0]):] = bits[:, :]

    return np.packbits(a, axis=1)


def exercise_02():
    pass


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
