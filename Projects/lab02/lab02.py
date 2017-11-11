#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import lab01.lab01 as lab01


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
    vj, tj = lab01.uniform_midrise_quantizer(vmax, delta_q)

    n = np.arange(0, 8)
    m = np.round(np.sin(2 * np.pi * (np.float(1300) / 8000) * n), decimals=3)

    mq, idx = lab01.quantize(m, vmax, vj, tj)

    bin = pcm_encode(idx, 3)
    dec = pcm_decode(bin)

    xq = dequantize(vj, dec)

    print('Quantized signal must be equal to (Encode > Decode > Dequantize): {}'. format(np.array_equal(mq, xq)))


def exercise_01():
    signal = lab01.sawtooth_signal()
    vmax = lab01.vmax_calculation(signal)
    r = 3

    delta_q = lab01.quantization_interval(vmax, r)
    vj, tj = lab01.uniform_midrise_quantizer(vmax, delta_q)
    mq, idx = lab01.quantize(signal, vmax, vj, tj)

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

    vmax = lab01.vmax_calculation(m)
    r = np.array([3, 5, 8])
    snr = np.arange(len(r), dtype='float')

    for i in range(len(r)):
        delta_q = lab01.quantization_interval(vmax, r[i])
        vj, tj = lab01.uniform_midrise_quantizer(vmax, delta_q)
        mq, idx = lab01.quantize(m, vmax, vj, tj)

        filename = 'lab02/som_8_16_quantize_{}bits.wav'.format(r[i])
        wav.write(filename, fs, mq.astype('int16'))

        bin = pcm_encode(idx, r[i])
        dec = pcm_decode(bin)

        filename = 'lab02/som_8_16_quantize_encode_decode_dequantize_{}bits.wav'.format(r[i])
        wav.write(filename, fs, dequantize(vj, dec).astype('int16'))

        p = np.sum(m * m) / len(m)
        snr[i] = lab01.snr_theoric(r[i], p, vmax)


def dequantize(vj, indexes):
    return vj[indexes]


def exercise_03():
    # Hamming parameters
    n = 15
    r = 11

    # Signal quantization
    m = lab01.sawtooth_signal()

    vmax = lab01.vmax_calculation(m)
    delta_q = lab01.quantization_interval(vmax, r)
    vj, tj = lab01.uniform_midrise_quantizer(vmax, delta_q)
    mq, idx = lab01.quantize(m, vmax, vj, tj)

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
    ], dtype='uint8')

    # Encode and codify with Hamming(15, 11)
    bin = pcm_encode(idx, r)
    x = hamming(bin, P, n, r)

    # Simulate channel communication
    y = channel(x, 0.01)

    # Measure the elapsed time to correct errors
    from time import time
    start_time = time()

    # Detect and correct error/noise
    y = error_correction(y, P)

    elapsed_time = time() - start_time
    print('Error correction elapsed time: {} s'.format(elapsed_time))


def channel(x, ber):
    x = np.copy(x)

    # Generate error/noise to simulate channel communication
    error = np.random.binomial(1, ber, len(x[0]) * len(x))
    y = (np.ndarray.flatten(x) + error) % 2

    # Build back the matrix from vector
    col = len(x[0])
    row = len(x)
    y = np.reshape(y, (row, col))

    return y


def hamming(message, P, n, r):
    G = np.hstack((np.identity(n - len(P[0]), dtype='uint8'), P))

    # C = np.logical_xor(m, G)
    return np.dot(message, G) % 2


def error_correction(message, P):
    # Form the H matrix
    H = np.vstack((P, np.identity(len(P[0]))))

    # Divide the message by r bits
    # col = len(H)
    # row = np.int32(len(message) / col)
    # message = np.reshape(message, (row, col))

    # Calculate the S matrix
    S = np.dot(message, H) % 2

    # If S is == 0 then there's no error in that sub-message
    # Else find the row position where sub-message is equal to S
    # Then flip the bit in the sub-message at row position
    for row in range(len(S)):
        if np.all(S[row] == 0):
            continue

        col = np.argwhere(np.all(H == S[row], axis=1))[0][0]
        message[row, col] = np.logical_not(message[row, col])

    return message[:, 0:-1 * len(P[0])]


def exercise_04():
    # Hamming parameters
    n = 15
    r = 11

    # Signal quantization
    m = lab01.sawtooth_signal()

    vmax = lab01.vmax_calculation(m)
    delta_q = lab01.quantization_interval(vmax, r)
    vj, tj = lab01.uniform_midrise_quantizer(vmax, delta_q)
    mq, idx = lab01.quantize(m, vmax, vj, tj)

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
    ], dtype='uint8')

    # Encode and codify with Hamming(15, 11)
    bin = pcm_encode(idx, r)
    x = hamming(bin, P, n, r)

    ber_theoric = np.array([.01, .05, .1, .5, .75, 1])
    ber_pratic = np.zeros(shape=(len(ber_theoric), 2))
    snr = np.zeros(len(ber_theoric))

    for i in range(len(ber_theoric)):
        # Channel simmulation
        y = channel(x, ber_theoric[i])
        ber_pratic[i, 0] = bit_error_rate(x, y)

        # Error correction
        y = error_correction(y, P)
        ber_pratic[i, 1] = bit_error_rate(bin, y)

        # Decode and Dequantize
        idx = pcm_decode(y)
        mq = dequantize(vj, idx)

        # SNR
        error = m - mq
        p_error = lab01.signal_power(error)
        p = lab01.signal_power(m)

        snr[i] = lab01.snr_pratic(p, p_error)


def bit_error_rate(a, b):
    return np.sum(np.array([a != b])) / len(a)


def exercise_05():
    f = 3500
    t = np.linspace(0, 8, f * 8)
    m = np.sin(2 * np.pi * f * t)
    fs = 8000

    if not nyquist(f, fs):
        return

    m = m[0:4]
    r = 3

    vmax = lab01.vmax_calculation(m)
    delta_q = lab01.quantization_interval(vmax, r)
    vj, tj = lab01.uniform_midrise_quantizer(vmax, delta_q)
    mq, idx = lab01.quantize(m, vmax, vj, tj)
    x = pcm_encode(idx, r)

    print('')


def nyquist(f, fs):
    return f * 2 <= fs


def exercise_06():
    pass


if __name__ == '__main__':
    main()
