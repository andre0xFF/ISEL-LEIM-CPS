#!/usr/bin/env python3.6
import lib.digital_modulation as modulation
import lib.channel as channel
import lib.quantization
import lib.quantization as quantization
import lib.error_control as error_control
import lib.codification as codification
import lib.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt


def main():
    exercise_05()
    exercise_06()
    exercise_07()


def exercise_05():
    msg = np.array([[0, 1, 0, 1, 1, 1]])
    a = 1
    p = 8

    x = modulation.manchester_enconde(msg, a, p)

    sigma = 1
    y1 = channel.send_with_awgn(x, sigma=np.sqrt(sigma))

    y2 = modulation.machester_decode(y1, 0, p)

    print('3. a) Bits at the emitter: {}'.format(msg))
    print('3. a) Line code (P = {}, A = {}): {}'.format(p, a, x))
    print('3. a) Channel transmission with AWGN: {}'.format(y1))
    print('3. a) Adapted filter result (Sigma = {}): {}'.format(sigma, y2))

    sigma = np.array([0.5, 1, 2])

    for s in sigma:
        y1 = channel.send_with_awgn(x, s)
        y2 = modulation.machester_decode(y1, 0, p)

        print('3. b) Adapted filter result (Sigma = {}): {}'.format(s, y2))

    print('3. b) The sigma value can amplify the amount of noise, therefore the adapted filter output'
          'may contain errors. In our example we have a higher error probability when sigma = 2.')


def exercise_06():
    # Hamming parameters
    n = 15
    r = 11

    # Quantization
    from lab01 import lab01
    m1 = lab01.sawtooth_signal()

    vmax = quantization.vmax(m1)
    delta_q = quantization.delta_q(vmax, r)
    vj, tj = quantization.uniform_midrise_quantizer(vmax, delta_q)
    x1, idx = quantization.quantize(m1, vmax, vj, tj)

    # Encoder
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

    x2 = codification.pcm_encode(idx, r)

    # Error control
    x3 = error_control.hamming(x2, P, n, r)

    # Digital modulation
    a = 1
    x4 = modulation.manchester_enconde(x3, a)

    # Channel
    sigma_squared = np.array([0.5, 1, 2, 4])

    for s in sigma_squared:
        y1 = channel.send_with_awgn(x4, np.sqrt(s))

        # Digital modulation
        y2 = modulation.machester_decode(y1, lambda_=0)

        # Error control
        y3 = error_control.correction(y2, P)

        # BER calculation
        # TODO: tb = len(x4[0]) ?
        tb = len(x4)
        eb = metrics.eb_manchester(a, tb)
        n0 = s * 2

        ber_pratic = metrics.ber(x3, y2)
        ber_theoric = metrics.ber_manchester(eb, n0)

        # Decoder
        y4 = codification.pcm_decode(y3)

        # Quantization
        m2 = quantization.dequantize(vj, y4)

        # Metrics
        # TODO: SNR?
        plt.plot(m1)
        plt.plot(m2)
        plt.show()

    # Metrics
    # snr = np.zeros(len(sigma_squared))

    px = metrics.signal_power(m1)
    snr_theoric = lib.metrics.snr_theoric(r, px, vmax)

    error = m1 - x1
    pe = metrics.signal_power(error)
    snr_pratic = lib.metrics.snr_db(px, pe)

    print('')


def exercise_07():
    # TODO
    roll_off = 0.5
    no = 0.5 * (1 / np.power(10, 6)) * 2
    att = 5
    bt = 400 * 1000
    ber_theoric = 1 / np.power(10, 5)

    rb = bt / (1 + roll_off)

    print('7. a) Max bit rate: {:8.2f} bits/s'.format(rb))

    tb = 1 / rb
    a = np.sqrt(no)

    eb = a * tb

    print('7. b) Energy per bit: {}'.format(eb))


if __name__ == '__main__':
    main()
