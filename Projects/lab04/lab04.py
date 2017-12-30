import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from lib import codification, error_control, quantization, digital_modulation, channel, metrics


def main():
    # exercise()
    constellation_graph()


def test_wav():
    filename = 'lab01/som_8_16_mono.wav'
    fs, m = wav.read(filename)
    return m


def super_mario_intro(variation_name):
    filename = 'Super-Mario-Bros-Music-Ground-Theme_{}.wav'.format(variation_name)
    fs, m = wav.read(filename)
    return np.mean(m, axis=1), fs


def sawtooth_signal():
    from lab01 import lab01
    m1 = lab01.sawtooth_signal()
    return m1


def exercise():
    out_filename = 'lab04/super_mario_intro_noise.wav'

    # hamming parameters
    n = 15
    r = 11

    # quantization
    m1, fs = super_mario_intro('3sec')
    m1 = sawtooth_signal()

    vmax = quantization.vmax(m1)
    delta_q = quantization.delta_q(vmax, r)
    vj, tj = quantization.uniform_midrise_quantizer(vmax, delta_q)

    x1, idx = quantization.quantize(m1, vmax, vj, tj)

    # codification
    x2 = codification.pcm_encode(idx, r)

    # error control
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

    # digital modulation
    x4, new_bits = digital_modulation.qam_encode(x3, p=8)

    # channel
    sigma_squared = np.array([0.05, 0.1, 0.15, 0.2])

    received_signals = np.zeros(shape=(len(m1), 4))

    snr_channel = np.zeros(len(sigma_squared))
    snr_reception = np.zeros(len(sigma_squared))

    ber_bc = np.zeros(len(sigma_squared))
    ber_ac = np.zeros(len(sigma_squared))

    for i in range(len(sigma_squared)):
        y1 = channel.send_with_awgn(x4, sigma=np.sqrt(sigma_squared[i]))

        # digital modulation
        y2 = digital_modulation.qam_decode(y1, p=8, rm_bits=new_bits)

        # error control
        y3 = error_control.correction(y2, parity_matrix)

        # codification
        y4 = codification.pcm_decode(y3)

        # quantization
        m2 = quantization.dequantize(vj, y4)

        # output
        wav.write(out_filename, fs, m2.astype('int16'))

        # metrics
        received_signals[:, i] = m2

        p_x4 = metrics.signal_power(x4)
        p_y1 = metrics.signal_power(y1)
        snr_channel[i] = metrics.snr(p_x4, p_y1)

        p_m1 = metrics.signal_power(m1)
        p_m2 = metrics.signal_power(m2)
        snr_reception[i] = metrics.snr(p_m1, p_m2)

        ber_bc[i] = metrics.ber(x3, y2)

    # metrics
    plt.title('Transmitted signal')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.plot(m1)
    plt.show()

    plt.figure(figsize=(12, 10))
    plt.suptitle('Received signal')

    for i in range(len(sigma_squared)):
        plt.subplot(2, 2, i + 1)
        plt.plot(received_signals[:, i])
        plt.title('AWGN ' r'$\sigma={}$'.format(sigma_squared[i]))
        plt.xlabel('time')
        plt.ylabel('amplitude')

    plt.tight_layout()
    plt.show()


def constellation_graph():
    cell_core_x = np.arange(-3, 4, 2)
    cell_core_x = np.hstack((cell_core_x, cell_core_x, cell_core_x, cell_core_x))

    cell_core_y = np.hstack((np.ones(4), np.ones(4) * 3, np.ones(4) * -1, np.ones(4) * -3))

    plt.scatter(cell_core_x, cell_core_y, s=10)

    line_x = np.array([-4, 4, 0, 0, -4, 4, -4, 4, -2, -2, 2, 2])
    line_y = np.array([0, 0, -4, 4, 2, 2, -2, -2, -4, 4, -4, 4])

    for i in range(0, len(line_x), 2):
        plt.plot([line_x[i], line_x[i + 1]], [line_y[i], line_y[i + 1]], '--', linewidth=1, color=(0.5, 0.5, 0.5))

    point_x = np.array([0.3,-1.2,2.3,-2.6,2.7])
    point_y = np.array([2.3,0.2,-1.3,2.6,-0.7])

    plt.scatter(point_x, point_y, s=3)

    plt.show()


if __name__ == '__main__':
    main()
