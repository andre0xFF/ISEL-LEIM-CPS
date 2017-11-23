#!/usr/bin/env python3.6
import lib.modulation as modulation
import lib.channel as channel
import lib.quantization as quantization
import lib.error_control as error_control
import lib.codification as codification
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
    y1 = channel.send_with_awgn(x, sigma)

    y2 = modulation.adapted_filter(y1, 0, p)

    print('3. a) Bits at the emitter: {}'.format(msg))
    print('3. a) Line code (P = {}, A = {}): {}'.format(p, a, x))
    print('3. a) Channel transmission with AWGN: {}'.format(y1))
    print('3. a) Adapted filter result (Sigma = {}): {}'.format(sigma, y2))

    sigma = np.array([0.5, 1, 2])

    for s in sigma:
        y1 = channel.send_with_awgn(x, s)
        y2 = modulation.adapted_filter(y1, 0, p)

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
    delta_q = quantization.interval(vmax, r)
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
    x4 = modulation.manchester_enconde(x3, a=1)

    # Channel
    sigma = np.array([0.5, 1, 2, 4])

    for s in sigma:
        y1 = channel.send_with_awgn(x4, sigma=s)

        # Digital modulation
        y2 = modulation.adapted_filter(y1, lambda_=0)

        # Error control
        y3 = error_control.correction(y2, P)

        # Decoder
        y4 = codification.pcm_decode(y3)

        # Quantization
        m2 = quantization.dequantize(vj, y4)

        plt.plot(m1)
        plt.plot(m2)
        plt.show()


def exercise_07():
    pass


if __name__ == '__main__':
    main()
