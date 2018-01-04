import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from lib import codification, error_control, quantization, digital_modulation, channel, metrics


def main():
    exercise()


def test_wav():
    filename = 'lab01/som_8_16_mono.wav'
    fs, m = wav.read(filename)
    return m, fs, 'som_8_16_mono_sigma_{}.wav'


def super_mario_intro(variation_name):
    filename = 'Super-Mario-Bros-Music-Ground-Theme_{}.wav'.format(variation_name)
    fs, m = wav.read(filename)
    return np.mean(m, axis=1), fs, 'Super-Mario-Bros-Music-Ground-Theme_sigma_{}.wav'


def sawtooth_signal():
    from lab01 import lab01
    m1 = lab01.sawtooth_signal()
    return m1, 0, 'sawtooth_{}.wav'


def exercise():
    # hamming parameters
    n = 15
    r = 11

    # quantization
    m1, fs, filename = super_mario_intro('3sec')
    # m1, fs, filename = test_wav()
    # m1, fs, filename = sawtooth_signal()

    filename = 'lab04/{}'.format(filename)

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
    x4, coords_o, new_bits = digital_modulation.qam_encode(x3, p=8)

    # channel
    sigma_square = np.array([0.05, 0.1, 0.2, 0.3])

    snr_channel = np.zeros(len(sigma_square))
    snr_reception = np.zeros(len(sigma_square))

    ber_bc = np.zeros(len(sigma_square))
    ber_ac = np.zeros(len(sigma_square))

    fig_signal = plt.figure(1)
    fig_received_signal = plt.figure(2, figsize=(12, 10))
    fig_constellation = plt.figure(3, figsize=(12, 10))

    for i in range(len(sigma_square)):
        y1 = channel.send_with_awgn(x4, sigma=np.sqrt(sigma_square[i]))

        # digital modulation
        y2, coords_r, coords_p = digital_modulation.qam_decode(y1, p=8, rm_bits=new_bits)

        # error control
        y3 = error_control.correction(y2, parity_matrix)

        # codification
        y4 = codification.pcm_decode(y3)

        # quantization
        m2 = quantization.dequantize(vj, y4)

        # output
        wav.write(filename.format(sigma_square[i]), fs, m2.astype('int16'))

        # metrics
        ax_received_signal = fig_received_signal.add_subplot(2, 2, i + 1)
        ax_constellation = fig_constellation.add_subplot(2, 2, i + 1)

        signal_graph(ax_received_signal, m2, sigma_square[i])
        constellation_graph(ax_constellation, coords_o, coords_p, coords_r, sigma_square[i])

        p_x4 = metrics.signal_power(x4)
        p_y1 = metrics.signal_power(y1)
        snr_channel[i] = metrics.snr(p_x4, p_y1)

        p_m1 = metrics.signal_power(m1)
        p_m2 = metrics.signal_power(m2)
        snr_reception[i] = metrics.snr(p_m1, p_m2)

        ber_bc[i] = metrics.ber(x3, y2)
        ber_ac[i] = metrics.ber(x2, y3)

    # metrics
    ax_signal = fig_signal.add_subplot(1, 1, 1)
    ax_signal.set_title('Transmitted signal')
    ax_signal.set_xlabel('time')
    ax_signal.set_ylabel('amplitude')
    ax_signal.plot(m1)

    fig_received_signal.suptitle('Received signal')
    fig_constellation.suptitle('16-QAM constellation')

    fig_ber_snr = plt.figure(4)
    ber_snr_graph(fig_ber_snr.add_subplot(1, 1, 1), ber_ac, ber_bc, snr_channel, snr_reception, sigma_square)

    plt.tight_layout()
    plt.show()


def ber_snr_graph(ax, ber_ac: np.ndarray, ber_bc: np.ndarray, snr_channel: np.ndarray, snr_reception: np.ndarray, sigma: np.ndarray):
    ax.plot(sigma, ber_ac, label='BER ac')
    ax.plot(sigma, ber_bc, label='BER bc')
    ax.plot(sigma, snr_channel, label='SNR channel')
    ax.plot(sigma, snr_reception, label='SNR reception')
    ax.set_xticks(sigma)
    ax.set_xlabel('AWGN ' r'$\sigma$')
    ax.legend()


def signal_graph(ax, m: np.ndarray, sigma: np.float):
    ax.plot(m)
    ax.set_title('AWGN ' r'$\sigma={}$'.format(sigma))
    ax.set_xlabel('time')
    ax.set_ylabel('amplitude')


def constellation_graph(ax, original_coords: np.ndarray, predicted_coords: np.ndarray, rounded_predicted_coords: np.ndarray, sigma: np.int):
    # cells' core
    cell_core_x = np.arange(-3, 4, 2)
    cell_core_x = np.hstack((cell_core_x, cell_core_x, cell_core_x, cell_core_x))
    cell_core_y = np.hstack((np.ones(4), np.ones(4) * 3, np.ones(4) * -1, np.ones(4) * -3))

    ax.scatter(
        cell_core_x,
        cell_core_y,
        s=10, label=['cell core']
    )

    # cells' lines
    line_x = np.array([-4, 4, 0, 0, -4, 4, -4, 4, -2, -2, 2, 2])
    line_y = np.array([0, 0, -4, 4, 2, 2, -2, -2, -4, 4, -4, 4])

    for i in range(0, len(line_x), 2):
        ax.plot([line_x[i], line_x[i + 1]], [line_y[i], line_y[i + 1]], '--', linewidth=1, color=(0.5, 0.5, 0.5))

    # coordinates
    plt_coords = original_coords != rounded_predicted_coords

    for i in range(0, len(plt_coords[0]), 2):
        plt_coords[:, i] = np.logical_or(plt_coords[:, i], plt_coords[:, i + 1])
        plt_coords[:, i + 1] = plt_coords[:, i]

    # well classified coordinates
    ax.scatter(
        predicted_coords[np.logical_not(plt_coords)][0::2],
        predicted_coords[np.logical_not(plt_coords)][1::2],
        s=0.5, c=(0.9, 0.9, 0.9), alpha=0.05
    )

    # badly classified coordinates
    ax.scatter(
        predicted_coords[plt_coords][0::2],
        predicted_coords[plt_coords][1::2],
        s=0.5, c=(1, 0, 0)
    )

    ax.set_xticks(np.arange(-4, 5))
    ax.set_yticks(np.arange(-4, 5))
    ax.set_title(r'$\sigma={}$'.format(sigma))


def constellation_configuration():
    cell_core_x = np.arange(-3, 4, 2)
    cell_core_x = np.hstack((cell_core_x, cell_core_x, cell_core_x, cell_core_x))
    cell_core_y = np.hstack((np.ones(4), np.ones(4) * 3, np.ones(4) * -1, np.ones(4) * -3))
    labels = ['1001', '1000', '0000', '0001', '1011', '1010', '0010', '0011', '1101', '1100', '0100', '0101', '1111',
              '1110', '0110', '0111']

    for i in range(len(cell_core_x)):
        plt.scatter(
            cell_core_x[i], cell_core_y[i],
            marker=r'${}$'.format(labels[i]), linestyle='None', s=1000, c=(0, 0, 0)
        )

    line_x = np.array([-4, 4, 0, 0, -4, 4, -4, 4, -2, -2, 2, 2])
    line_y = np.array([0, 0, -4, 4, 2, 2, -2, -2, -4, 4, -4, 4])

    for i in range(0, len(line_x), 2):
        plt.plot([line_x[i], line_x[i + 1]], [line_y[i], line_y[i + 1]], '--', linewidth=1, color=(0.5, 0.5, 0.5))

    plt.show()


if __name__ == '__main__':
    main()
