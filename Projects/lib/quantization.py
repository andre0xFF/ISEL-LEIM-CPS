import numpy as np


# r: number of bits per sample
# vmax: max value to quantify
# type-: quantifier-type (midrise or midtread)
# vj: values of quantization
# tj: values of decision
def uniform_midtread_quantizer(vmax, delta_q):
    vj = np.arange(-1 * vmax + delta_q, vmax + delta_q / 2, delta_q)
    tj = np.arange(-1 * vmax + delta_q * (3 / 2), vmax, delta_q)

    return vj, tj


def uniform_midrise_quantizer(vmax, delta_q):
    vj = np.arange(-1 * vmax + delta_q / 2, vmax, delta_q)
    tj = np.arange(-1 * vmax + delta_q, vmax, delta_q)

    return vj, tj


def interval(v, r):
    return (2 * v) / (np.power(2, r))


def vmax(signal: np.ndarray) -> np.float:
    return np.max(np.abs(signal))


# mq: quantified signal
# idx: indexes of each quantified value
def quantize(signal, vmax, vj, tj):
    tj = np.insert(tj, len(tj), vmax)

    # Majorate mq array as default value
    mq = np.ones(len(signal)) * np.max(vj)

    # Majorate the index as default value
    # idx = np.ones(shape=(len(signal), 1), dtype='uint8') * len(vj)
    idx = np.ones(len(signal), dtype='uint32') * len(vj)

    # Loop every point in the signal and check if the point is lower or equal
    # than any tj elements (decision values)
    for i in range(len(signal)):
        eval = signal[i] <= tj

        # Test whether any array element along a given axis evaluates to True
        if np.any(eval):
            xq_value = vj[eval][0]
            mq[i] = xq_value

            # Get the index
            k = np.nonzero(eval)[0][0]
            idx[i] = k

    return mq, idx


def dequantize(vj, indexes):
    return vj[indexes]


# r: number of bits
# p: signal power
# v: vmax
def snr_theoric(r: np.int, p: np.float, vmax: np.float) -> np.float:
    return 6.02 * r + 10 * np.log10(3 * p / np.power(vmax, 2))


def snr_pratic(p_signal: np.float, p_quantized: np.float) -> np.float:
    return 10 * np.log10(p_signal / p_quantized)