#!/usr/bin/env python3.6
import numpy as np
import scipy.io.wavfile as wav
import lab01.lab01 as lab01
import lib.quantization as q
import lib.codification as c
import lib.metrics as metrics
import lib.error_control as error_control


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
    vj, tj = q.uniform_midrise_quantizer(vmax, delta_q)

    n = np.arange(0, 8)
    m = np.round(np.sin(2 * np.pi * (np.float(1300) / 8000) * n), decimals=3)

    mq, idx = q.quantize(m, vmax, vj, tj)

    bin = c.pcm_encode(idx, 3)
    dec = c.pcm_decode(bin)

    xq = q.dequantize(vj, dec)

    # print('Quantized signal must be equal to (Encode > Decode > Dequantize): {}'. format(np.array_equal(mq, xq)))


def exercise_01():
    signal = lab01.sawtooth_signal()
    vmax = q.vmax(signal)
    r = 3

    delta_q = q.interval(vmax, r)
    vj, tj = q.uniform_midrise_quantizer(vmax, delta_q)
    mq, idx = q.quantize(signal, vmax, vj, tj)

    bin = c.pcm_encode(idx, r)
    dec = c.pcm_decode(bin)


def exercise_02():
    filename = 'lab01/som_8_16_mono.wav'
    print('2.    Reading sound file: {}'.format(filename))
    fs, m = wav.read(filename)

    vmax = q.vmax(m)
    r = np.array([3, 5, 8])
    snr = np.arange(len(r), dtype='float')

    for i in range(len(r)):
        delta_q = q.interval(vmax, r[i])
        vj, tj = q.uniform_midrise_quantizer(vmax, delta_q)
        mq, idx = q.quantize(m, vmax, vj, tj)

        filename = 'lab02/som_8_16_quantize_{}bits.wav'.format(r[i])
        print('2.    r = {} [quantize]; Writing sound file {}'.format(r[i], filename))
        wav.write(filename, fs, mq.astype('int16'))

        bin = c.pcm_encode(idx, r[i])
        dec = c.pcm_decode(bin)

        filename = 'lab02/som_8_16_quantize_encode_decode_dequantize_{}bits.wav'.format(r[i])
        print('2.    r = {} [quantize > encode > decode > dequantize]; Writing sound file {}'.format(r[i], filename))
        wav.write(filename, fs, q.dequantize(vj, dec).astype('int16'))

        p = np.sum(m * m) / len(m)
        snr[i] = metrics.snr_theoric(r[i], p, vmax)

        print('2.    r = {:d}; SNR = {:>7.3f}'.format(r[i], snr[i]))


def exercise_03():
    # Hamming parameters
    n = 15
    r = 11

    # Signal quantization
    m = lab01.sawtooth_signal()

    vmax = q.vmax(m)
    delta_q = q.interval(vmax, r)
    vj, tj = q.uniform_midrise_quantizer(vmax, delta_q)
    mq, idx = q.quantize(m, vmax, vj, tj)

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
    bin = c.pcm_encode(idx, r)
    x = error_control.hamming(bin, P, n, r)

    # Simulate channel communication
    y = channel(x, 0.01)

    # Measure the elapsed time to correct errors
    from time import time
    start_time = time()

    # Detect and correct error/noise
    y = error_control.correction(y, P)

    elapsed_time = time() - start_time
    print('3.    Error correction elapsed time: {:4.3f} s'.format(elapsed_time))


def channel(x, ber):
    x = np.copy(x)

    # Generate error/noise to simulate channel communication
    e = np.random.binomial(1, ber, len(x[0]) * len(x))
    y = (np.ndarray.flatten(x) + e) % 2

    # Build back the matrix from vector
    col = len(x[0])
    row = len(x)
    y = np.reshape(y, (row, col))

    return y


def exercise_04():
    # Parity matrix: http://michael.dipperstein.com/hamming/

    # Hamming parameters
    n = 15
    r = 11

    # Signal quantization
    m = lab01.sawtooth_signal()

    vmax = q.vmax(m)
    delta_q = q.interval(vmax, r)
    vj, tj = q.uniform_midrise_quantizer(vmax, delta_q)
    mq, idx = q.quantize(m, vmax, vj, tj)

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
    bin = c.pcm_encode(idx, r)
    x = error_control.hamming(bin, P, n, r)

    ber_theoric = np.array([.01, .05, .1, .5, .75, 1])
    ber_pratic = np.zeros(shape=(len(ber_theoric), 2))
    snr = np.zeros(len(ber_theoric))

    for i in range(len(ber_theoric)):
        # Channel simulation
        y = channel(x, ber_theoric[i])
        ber_pratic[i, 0] = error_control.bit_error_rate(x, y)

        # Error correction
        y = error_control.correction(y, P)
        ber_pratic[i, 1] = error_control.bit_error_rate(bin, y)

        # Decode and Dequantize
        idx = c.pcm_decode(y)
        mq = q.dequantize(vj, idx)

        # SNR
        e = m - mq
        p_error = metrics.signal_power(e)
        p = metrics.signal_power(m)

        snr[i] = metrics.snr_pratic(p, p_error)

        print('4.    BERt = {:>6.2f}; BER = {:>6.3f}; BER\' = {:>6.3f}; SNR = {:>6.3f}'.format(
            ber_theoric[i], ber_pratic[i, 0], ber_pratic[i, 1], snr[i])
        )


def exercise_05():
    # TODO
    # a)
    fs = 8000
    f = 3500

    t = np.arange(0, 5 * 1 / f, 1 / fs)
    m = np.sin(2 * np.pi * f * t)

    r = 3

    vmax = q.vmax(m)
    delta_q = q.interval(vmax, r)
    vj, tj = q.uniform_midrise_quantizer(vmax, delta_q)
    mq, idx = q.quantize(m, vmax, vj, tj)
    x = c.pcm_encode(idx, r)

    print('5. a) First 4 samples of the signal y(t) = sin(2 * pi * 3500 * t) with fs = {}: {}'.format(fs, m))
    print('5. a) Midrise quantizer with r = 3 codification: {}'.format(np.ndarray.flatten(x)))


def exercise_06():
    block = np.array([1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1])
    key = np.array([1, 0, 0, 1, 1])

    msg = crc(block, key)
    print('6. a) Message generated by the Hamming code H(15, 11) (x^4 + x + 1): {}'.format(msg))

    print('6. b) The received message {} is equal to the transmitted message generated by the Hamming code'
          'H(15, 11) (x^4 + x + 1). Therefore there is no error.'.format(np.append(block, key)))

    n = 15
    k = 11
    ber_initial = 1 / np.power(10, 3)
    ber_line = error_control.ber_after_error_correction(ber_initial, n)

    print('6. c) BER\' = {:9.7f}'.format(ber_line))

    rb = 10 * np.power(10, 6)
    te_before = metrics.time_between_errors(rb, ber_initial)
    te_after = metrics.time_between_errors(rb, ber_line)

    print('6. d) Te = {:7.5f} (before errors correction)'.format(te_before))
    print('6. d) Te = {:7.5f} (after errors correction)'.format(te_after))


def crc(m, key):
    # http://www.geeksforgeeks.org/modulo-2-binary-division/
    mx = np.append(m, np.zeros(len(key) - 1, dtype='uint8'))
    r = mod2div(mx, key)
    r = r[1:len(r)]

    return np.append(m, r)


def mod2div(dividend, divisor):
    digit = len(divisor)
    block = dividend[0:digit]

    while digit < len(dividend):

        if block[0]:
            block = np.bitwise_xor(divisor, block)
        else:
            block = np.bitwise_xor(np.zeros(len(block), dtype='uint8'), block)

        block = block[1:len(block)]
        block = np.append(block, dividend[digit])
        digit += 1

    if block[0]:
        block = np.bitwise_xor(divisor, block)
    else:
        block = np.bitwise_xor(np.zeros(len(block), dtype='uint8'), block)

    return block


if __name__ == '__main__':
    main()
