import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from lib import codification, error_control, quantization, digital_modulation, channel


def main():
    exercise()
    # wav
    # ber, snr
    # grafico aumentando a potencia do ruido e calculando o ber pratico e teorico / snr


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

    # Hamming parameters
    n = 15
    r = 11

    # Quantization
    m1, fs = super_mario_intro('3sec')

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
    sigma = np.array([0.05, 0.1, 0.15, 0.2])
    received_signals = np.zeros(shape=(len(m1), 4))

    for i in range(len(sigma)):
        y1 = channel.send_with_awgn(x4, sigma=np.sqrt(sigma[i]))

        # Digital modulation
        y2 = digital_modulation.qam_decode(y1, p=8, rm_bits=new_bits)

        # Error control
        y3 = error_control.correction(y2, parity_matrix)

        # Codification
        y4 = codification.pcm_decode(y3)

        # Quantization
        m2 = quantization.dequantize(vj, y4)

        wav.write(out_filename, fs, m2.astype('int16'))

        received_signals[:, i] = m2

    plt.title('Transmitted signal')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.plot(m1)
    plt.show()

    plt.figure(figsize=(12, 10))
    plt.suptitle('Received signal')

    for i in range(len(sigma)):
        plt.subplot(2, 2, i + 1)
        plt.plot(received_signals[:, i])
        plt.title('AWGN ' r'$\sigma={}$'.format(sigma[i]))
        plt.xlabel('time')
        plt.ylabel('amplitude')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
