import numpy as np

from lib import codification, error_control, quantization, digital_modulation, channel


def main():
    test()


def test():
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

    x4, new_bits = digital_modulation.qam_encode(x3, p=8)
    x5 = digital_modulation.qam_decode(x4, p=8, rm_bits=new_bits)

    y1 = channel.send_with_awgn(x4, sigma=np.sqrt(0.5))

    # ###

    x = np.array([[0,0,0,0],[1,0,1,1]])
    x4, new_bits = digital_modulation.qam_encode(x, p=8)
    x5 = digital_modulation.qam_decode(x4, p=8, rm_bits=new_bits)

    print()


if __name__ == '__main__':
    main()
