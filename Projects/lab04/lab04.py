import numpy as np
import matplotlib.pyplot as plt
from lib import codification, error_control, quantization, digital_modulation, channel


def main():
    exercise()


def exercise():
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

    # Codification
    x2 = codification.pcm_encode(idx, r)

    # Error control
    parity_matrix = np.array([
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

    x3 = error_control.hamming(x2, parity_matrix, n, r)

    # Digital modulation
    x4, new_bits = digital_modulation.qam_encode(x3, p=8)

    # Channel
    y1 = channel.send_with_awgn(x4, sigma=np.sqrt(0.2))

    # Digital modulation
    y2 = digital_modulation.qam_decode(y1, p=8, rm_bits=new_bits)

    # Error control
    y3 = error_control.correction(y2, parity_matrix)

    # Codification
    y4 = codification.pcm_decode(y3)

    # Quantization
    m2 = quantization.dequantize(vj, y4)

    plt.plot(m1)
    plt.show()

    plt.plot(m2)
    plt.show()


if __name__ == '__main__':
    main()
