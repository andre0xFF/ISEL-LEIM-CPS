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


def delta_q(vmax, r):
    return (2 * vmax) / (np.power(2, r))


def vmax(signal: np.ndarray) -> np.float:
    return np.max(np.abs(signal))


# mq: quantified signal
# idx: indexes of each quantified value
def quantize(signal, vmax, vj, tj):
    tj = np.insert(tj, len(tj), vmax)

    # majorate mq array as default value
    mq = np.ones(len(signal)) * np.max(vj)

    # majorate the index as default value
    # idx = np.ones(shape=(len(signal), 1), dtype='uint8') * len(vj)
    idx = np.ones(len(signal), dtype='uint32') * len(vj)

    # loop every point in the signal and check if the point is lower or equal
    # than any tj elements (decision values)
    for i in range(len(signal)):
        eval = signal[i] <= tj

        # test whether any array element along a given axis evaluates to True
        if np.any(eval):
            xq_value = vj[eval][0]
            mq[i] = xq_value

            # get the index
            k = np.nonzero(eval)[0][0]
            idx[i] = k

    return mq, idx


def dequantize(vj, indexes):
    return vj[indexes]
